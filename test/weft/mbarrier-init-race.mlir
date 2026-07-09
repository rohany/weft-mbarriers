// Initialization by a single thread with no synchronization (syncthreads)
// before other threads use the barrier. The simulated schedule happens to run
// the initializing thread first, so the simulation itself succeeds -- but
// nothing orders thread 1's wait after the initialization, so a different
// schedule can reach the wait on an uninitialized barrier and err. The
// initialization check (okInit) rejects the program.
//
// RUN: not weft standard %s 2>&1 | FileCheck %s
// CHECK: weft: initialization of mbarrier 0 (thread 0) does not happen-before use (thread 1)
// CHECK: weft: the program is not well-synchronized
module {
  func.func @init_race(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %false = arith.constant 0 : i1
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      barrier.mbarrier_init 0, 1
      // BUG: no syncthreads between the initialization and the uses below.
      barrier.smem_write %c0 : index
      barrier.mbarrier_arrive 0
    } else {
      barrier.mbarrier_wait 0 %false
      barrier.smem_read %c0 : index
    }
    return
  }
}
