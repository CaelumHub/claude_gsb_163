# -*- coding: utf-8 -*-
"""
调试器（Debugger）。

在 VM 的可续跑执行循环之上实现交互式调试：断点、单步进入、单步跳过、跳出、
继续执行、变量查看与调用栈快照。

核心难点"调试器与解释器的状态同步"体现在这里——VM 是同步、可挂起/续跑的，
调试器通过 VM.run(pause_fn, on_pause) 在**每条指令执行前**决定是否暂停，
暂停后帧栈 / 指令指针 / 局部变量 / 操作数栈 / 堆全部原样保留，等待下一次请求继续。

步进语义（以"下一个待执行指令"为判断对象）：
  * step_instruction  —— 执行恰好一条指令后暂停（最细粒度）；
  * step_into         —— 执行到下一个源码行（进入被调用函数的第一行）；
  * step_over         —— 执行到当前帧的下一个源码行（跳过函数调用内部）；
  * step_out          —— 执行到当前帧返回（回到上一帧）；
  * continue          —— 执行到下一个断点或程序结束。
"""

from typing import List, Optional, Set

from . import diagnostics as diag


# 暂停原因
PAUSE_BREAKPOINT = "breakpoint"
PAUSE_STEP = "step"
PAUSE_ENTRY = "entry"        # 尚未开始执行（初次启动）
PAUSE_FINISHED = "finished"
PAUSE_ERROR = "error"


class Debugger:
    def __init__(self, vm, breakpoints: Optional[Set[int]] = None):
        self.vm = vm
        vm.debugger = self
        self.breakpoints: Set[int] = set(breakpoints or [])
        self.step_mode: Optional[str] = None   # None | 'instruction' | 'into' | 'over' | 'out'
        self._step_from_line = 0
        self._step_depth = 0
        self._arm_pending = False   # 续跑后首次判定：放行当前位置的这一条指令
        self._ignore_line: Optional[int] = None  # 续跑后暂时忽略的断点行（离开即解除）
        self._at_entry = False      # 是否停在"入口"（首条指令尚未执行）
        self.pause_reason: str = PAUSE_ENTRY
        self._just_started = True

    # ------------------------------------------------------------------
    # 断点管理
    # ------------------------------------------------------------------
    def set_breakpoints(self, lines: List[int]):
        self.breakpoints = set(l for l in lines if isinstance(l, int) and l > 0)

    def add_breakpoint(self, line: int):
        self.breakpoints.add(line)

    def remove_breakpoint(self, line: int):
        self.breakpoints.discard(line)

    def clear_breakpoints(self):
        self.breakpoints.clear()

    def breakpoint_lines(self) -> List[int]:
        return sorted(self.breakpoints)

    # ------------------------------------------------------------------
    # 暂停判定（在每条指令执行前被 VM 调用）
    #
    # 返回值：True=在此暂停；False=继续；"arm"=本次先放行当前指令，
    # 从下一条指令起恢复正常判定。
    #
    # 断点去重：在某行暂停后续跑，_ignore_line 会暂时记下该行——同一源码行
    # 可能对应多条字节码指令，若不忽略会在续跑瞬间又命中同一断点；一旦执行
    # 离开该行（或跳到别的帧）即解除，循环重新回到该行时照常命中。
    # ------------------------------------------------------------------
    def should_pause(self, vm):
        ins = vm.peek_instruction()
        if ins is None:
            return False

        # 续跑后的第一次回调：先放行当前位置这一条指令
        if self._arm_pending:
            self._arm_pending = False
            return "arm"

        # 维护"暂时忽略的断点行"：离开该行即解除
        if self._ignore_line is not None and ins.line != self._ignore_line:
            self._ignore_line = None

        if self.step_mode == "instruction":
            self.pause_reason = PAUSE_STEP
            return True
        if self.step_mode == "into":
            if ins.line != self._step_from_line:
                self.pause_reason = PAUSE_STEP
                return True
            return False
        if self.step_mode == "over":
            # 回到不深于发起步进时的帧、且已推进到另一个源码行时暂停
            # （同一深度覆盖"跳过函数调用后返回当前帧"的情形；
            #   更浅的深度覆盖被调函数一路返回到上层的情形）
            if len(vm.frames) <= self._step_depth and ins.line != self._step_from_line:
                self.pause_reason = PAUSE_STEP
                return True
            return False
        if self.step_mode == "out":
            if len(vm.frames) < self._step_depth:
                self.pause_reason = PAUSE_STEP
                return True
            return False

        # 断点模式（step_mode is None）：精确匹配指令所属源码行
        if ins.line in self.breakpoints and ins.line != self._ignore_line:
            self.pause_reason = PAUSE_BREAKPOINT
            return True
        return False

    def on_pause(self, vm):
        """暂停发生后清除步进模式与续跑忽略状态。"""
        self.step_mode = None
        self._arm_pending = False
        self._ignore_line = None

    # ------------------------------------------------------------------
    # 交互命令
    # ------------------------------------------------------------------
    def start(self):
        """开始执行：若有断点则运行到第一个断点，否则运行到结束。"""
        if self._just_started:
            self._just_started = False
            self.vm.start()
            # 若第一行就有断点，暂停在入口（首条指令尚未执行）
            if self._first_line_breakpoint():
                self.pause_reason = PAUSE_BREAKPOINT
                self.vm.paused = True
                self._at_entry = True
                return
        self.continue_()

    def _first_line_breakpoint(self) -> bool:
        ins = self.vm.peek_instruction()
        return ins is not None and ins.line in self.breakpoints

    def _sync_finish_reason(self):
        """run 返回后若未暂停，说明程序已结束（正常结束或运行时错误）。"""
        if self.vm.paused:
            return
        if self.vm.error is not None:
            self.pause_reason = PAUSE_ERROR
        elif self.vm.finished:
            self.pause_reason = PAUSE_FINISHED

    def continue_(self):
        self.step_mode = None
        # 续跑总是先放行当前这条指令；但入口断点时不抑制该行——首条指令执行后
        # 若控制流仍落在该行（如循环回边），断点仍应正常生效。
        from_entry = self._at_entry
        self._at_entry = False
        self._arm_resume(suppress_line=not from_entry)
        self.vm.run(self.should_pause, self.on_pause)
        self._sync_finish_reason()

    def step_instruction(self):
        """单步一条指令（当前暂停位置的下一条指令执行后再停）。"""
        self.step_mode = "instruction"
        self._at_entry = False
        self._arm_resume(suppress_line=False)
        # arm 放行当前这一条；instruction 模式在下一条指令前即暂停
        self.vm.run(self.should_pause, self.on_pause)
        self.step_mode = None
        self._sync_finish_reason()

    def step_into(self):
        """单步到下一源码行（进入函数）。"""
        self._prepare_step("into")

    def step_over(self):
        """单步到当前帧的下一源码行（跳过函数）。"""
        self._prepare_step("over")

    def step_out(self):
        """执行到当前帧返回。"""
        self._prepare_step("out")

    def _arm_resume(self, suppress_line=True):
        """配置一次续跑：放行当前暂停位置的这一条指令（否则会在原地立刻再次
        暂停）。suppress_line=True 时，断点模式下还会暂时忽略当前暂停行，
        直到执行离开该行——避免同一源码行的多条指令让断点瞬间再次命中；
        入口断点续跑时传 False，使首条指令执行后若控制流仍在该行（如循环
        回边、或同行后续指令再次回到断点）仍能正常生效。
        """
        ins = self.vm.peek_instruction()
        self._arm_pending = True
        self._ignore_line = (
            ins.line if (suppress_line and ins is not None and self.step_mode is None) else None
        )
        self.vm.paused = False

    def _prepare_step(self, mode):
        ins = self.vm.peek_instruction()
        self._step_from_line = ins.line if ins else 0
        self._step_depth = len(self.vm.frames)
        from_entry = self._at_entry
        self._at_entry = False
        if from_entry and mode in ("into", "over"):
            # 入口首条指令尚未执行：第一个"步进"只执行一条指令便停下，
            # 不能一次跨过整个首行（首行往往对应多条指令）。instruction 模式
            # 会在放行首条后的下一条指令前暂停。
            self.step_mode = "instruction"
        else:
            self.step_mode = mode
        self._arm_resume(suppress_line=False)
        self.vm.run(self.should_pause, self.on_pause)
        self._sync_finish_reason()

    # ------------------------------------------------------------------
    # 状态快照
    # ------------------------------------------------------------------
    def snapshot(self):
        """生成调试暂停时的完整状态，供"执行跟踪/调用栈/变量监视"页面渲染。"""
        vm = self.vm
        ins = vm.peek_instruction()
        return {
            "finished": vm.finished,
            "paused": vm.paused,
            "reason": self.pause_reason,
            "instruction_count": vm.instruction_count,
            "elapsed_ms": round(vm.elapsed_ms(), 3),
            "error": vm.error.to_dict() if vm.error else None,
            "breakpoints": self.breakpoint_lines(),
            "next_instruction": ins.to_dict() if ins else None,
            "current_position": vm.current_position(),
            "call_stack": vm.frame_snapshot(),
            "operand_stack": vm.stack_snapshot(),
            "globals": {k: _serialize(vm, k) for k in sorted(vm.globals)},
            "output": list(vm.output),
            "return_value": _plain(vm.return_value),
        }

    def to_error_pause(self):
        self.pause_reason = PAUSE_ERROR
        self.step_mode = None


def _serialize(vm, name):
    from . import runtime as rt
    v = vm.globals[name]
    return rt.serialize_value(v)


def _plain(v):
    from . import runtime as rt
    if v is None:
        return None
    return rt.serialize_value(v)
