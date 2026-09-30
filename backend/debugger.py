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
        # 续跑"防自命中"状态：
        #   _suppress/_suppress_frame/_suppress_line 在暂停时记录"原暂停位置"；
        #   _hop_once 在发起继续/单步命令时置位，仅放行紧接着的第一条待执行指令
        #   （即触发本次暂停的那条）。之后只要执行仍停留在"原暂停帧的同一源码行"
        #   就继续放行；一旦该帧推进到下一行、跳入函数（帧变深）或从函数返回
        #   （帧变浅），即恢复暂停能力。
        # 一行内的多条字节码不会让断点自命中，而循环回边（先经过条件行）与函数
        # 的重复调用都能再次正常命中断点。
        self._hop_once = False
        self._suppress = False
        self._suppress_at_entry = False
        self._suppress_frame = None
        self._suppress_line = 0
        self.pause_reason: str = PAUSE_ENTRY
        self._just_started = True

    # ------------------------------------------------------------------
    # 断点管理
    # ------------------------------------------------------------------
    def set_breakpoints(self, lines: List[int]):
        new_bps = set(l for l in lines if isinstance(l, int) and l > 0)
        # 断点集合发生变化属于显式用户操作：清除续跑抑制状态，否则上一次暂停
        # 遗留的"同行放行"可能跳过用户新加在当前行上的断点
        if new_bps != self.breakpoints:
            self._suppress = False
            self._suppress_at_entry = False
            self._hop_once = False
        self.breakpoints = new_bps

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
    # ------------------------------------------------------------------
    def _arm_suppression(self, vm, at_entry=False):
        """记录暂停位置（帧 + 下一条待执行指令所在行），续跑时用于防自命中。

        at_entry=True 表示暂停在尚未执行任何指令的入口断点：此时该行的全部
        指令都还没执行过，续跑时应整行放行（不消费一次性放行），否则同一行
        的第二条指令又会立刻命中断点。
        """
        self._suppress = True
        self._suppress_at_entry = at_entry
        fr = vm.frames[-1] if vm.frames else None
        ins = vm.peek_instruction()
        self._suppress_frame = fr
        self._suppress_line = ins.line if ins else 0

    def _begin_resume(self):
        """发起继续 / 单步命令：下一条待执行指令（原暂停指令）放行一次。"""
        if getattr(self, "_suppress_at_entry", False):
            # 入口暂停：靠"同行抑制"放行整行，不需要额外的一次性放行
            self._hop_once = False
        else:
            self._hop_once = True

    def _still_suppressed(self, vm, ins) -> bool:
        if not self._suppress:
            return False
        # 续跑后的第一次检查对应原暂停指令本身，直接放行（入口暂停除外）
        if self._hop_once:
            self._hop_once = False
            return True
        fr = vm.frames[-1] if vm.frames else None
        # 帧已切换（进入被调函数 / 从函数返回），立即恢复暂停能力
        if fr is not self._suppress_frame:
            self._suppress = False
            self._suppress_at_entry = False
            return False
        # 仍在原帧且下一条指令属于原暂停行：一行对应多条字节码，继续放行；
        # 推进到别的源码行（顺序下一行 / 循环回边先经过条件行）则解除
        if ins is not None and ins.line == self._suppress_line:
            return True
        self._suppress = False
        self._suppress_at_entry = False
        return False

    def _stepping_suppressed(self, vm, ins) -> bool:
        """步进命令专用抑制：仅放行原暂停指令一次；帧一旦切换（函数调用/返回）
        立即解除，避免把"进入新函数的第一行"也误当作同一位置而放行。"""
        if not self._suppress:
            return False
        if self._hop_once:
            self._hop_once = False
            return True
        # 步进不做"同行多指令"放行：行级步进只看行号，指令级步进每条都停
        self._suppress = False
        self._suppress_at_entry = False
        return False

    def should_pause(self, vm) -> bool:
        ins = vm.peek_instruction()
        if ins is None:
            return False
        if self.step_mode == "instruction":
            # 放行原暂停指令恰好一次，随后无条件暂停（最细粒度，不按行合并）
            if self._stepping_suppressed(vm, ins):
                return False
            self.pause_reason = PAUSE_STEP
            return True
        if self.step_mode == "into":
            if self._stepping_suppressed(vm, ins):
                return False
            if ins.line != self._step_from_line:
                self.pause_reason = PAUSE_STEP
                return True
            return False
        if self.step_mode == "over":
            if self._stepping_suppressed(vm, ins):
                return False
            # 同帧换到下一行，或已从被调函数返回更浅帧，即暂停；进入更深帧不暂停
            if len(vm.frames) <= self._step_depth and ins.line != self._step_from_line:
                self.pause_reason = PAUSE_STEP
                return True
            return False
        if self.step_mode == "out":
            if self._stepping_suppressed(vm, ins):
                return False
            if len(vm.frames) < self._step_depth:
                self.pause_reason = PAUSE_STEP
                return True
            return False
        # 断点模式（step_mode is None）：抑制原暂停位置的自命中后正常判断
        if self._still_suppressed(vm, ins):
            return False
        if ins.line in self.breakpoints:
            self.pause_reason = PAUSE_BREAKPOINT
            return True
        return False

    def on_pause(self, vm):
        """暂停发生后：续跑时先抑制原暂停位置，清除步进模式。"""
        self._arm_suppression(vm, at_entry=False)
        self.step_mode = None
        self.pause_reason = vm.pause_reason if hasattr(vm, "pause_reason") and vm.pause_reason else self.pause_reason

    # ------------------------------------------------------------------
    # 交互命令
    # ------------------------------------------------------------------
    def start(self):
        """开始执行：若有断点则运行到第一个断点，否则运行到结束。"""
        if self._just_started:
            self._just_started = False
            self.vm.start()
            # 若第一行就有断点，暂停在入口；否则直接运行
            if self._first_line_breakpoint():
                self.pause_reason = PAUSE_BREAKPOINT
                # 入口断点：该行尚未执行，续跑时整行放行，避免立即自命中
                self._arm_suppression(self.vm, at_entry=True)
                self.vm.paused = True
                self.vm.pause_reason = PAUSE_BREAKPOINT
                return
        self.continue_()

    def _first_line_breakpoint(self) -> bool:
        ins = self.vm.peek_instruction()
        return ins is not None and ins.line in self.breakpoints

    def continue_(self):
        self.step_mode = None
        self._begin_resume()
        self.vm.paused = False
        self.vm.run(self.should_pause, self.on_pause)

    def step_instruction(self):
        """单步一条指令（放行当前暂停指令，下一条指令处暂停）。"""
        self.step_mode = "instruction"
        self._begin_resume()
        self.vm.paused = False
        self.vm.run(self.should_pause, self.on_pause)
        self.step_mode = None

    def step_into(self):
        """单步到下一源码行（进入函数）。"""
        self._prepare_step("into")

    def step_over(self):
        """单步到当前帧的下一源码行（跳过函数）。"""
        self._prepare_step("over")

    def step_out(self):
        """执行到当前帧返回。"""
        self._prepare_step("out")

    def _prepare_step(self, mode):
        ins = self.vm.peek_instruction()
        self._step_from_line = ins.line if ins else 0
        self._step_depth = len(self.vm.frames)
        self.step_mode = mode
        self._begin_resume()
        self.vm.paused = False
        self.vm.run(self.should_pause, self.on_pause)

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
            "reason": self.pause_reason if vm.paused or vm.finished else PAUSE_FINISHED,
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
