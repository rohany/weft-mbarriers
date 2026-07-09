// A use that precedes the barrier's initialization in program order errs
// (MB-Arrive-Err): the initializing thread itself reaches the arrive before
// executing the init.
//
// RUN: not weft standard %s 2>&1 | FileCheck %s
// CHECK: weft: arrive on uninitialized mbarrier 0
// CHECK: weft: failed to simulate the thread programs
module {
  func.func @use_before_init(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      // BUG: the arrive executes before the barrier is initialized.
      barrier.mbarrier_arrive 0
      barrier.mbarrier_init 0, 1
    }
    barrier.named_barrier_sync 0, 2
    return
  }
}
