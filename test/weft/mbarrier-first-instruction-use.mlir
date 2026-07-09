// A use of an mbarrier as the *first* instruction of a thread can never be
// ordered after the barrier's initialization: the thread can reach it at the
// initial configuration, where every mbarrier is uninitialized. The
// initialization check (okInit) rejects the program even though the simulated
// schedule (which runs thread 0's init first) succeeds.
//
// RUN: not weft standard %s 2>&1 | FileCheck %s
// CHECK: weft: use of mbarrier 0 (thread 1) is the first instruction of its thread
// CHECK: weft: the program is not well-synchronized
module {
  func.func @first_instruction_use(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      barrier.mbarrier_init 0, 2
      barrier.mbarrier_arrive 0
    } else {
      // After thread specialization and canonicalization this arrive is the
      // first instruction of thread 1's program.
      barrier.mbarrier_arrive 0
    }
    return
  }
}
