// Initializing an mbarrier twice is an error (MB-Init-Err): here every
// thread executes the init, so the second thread to reach it errs during
// simulation. (The checker also has a static unique-initialization check,
// okUniqueInit, but with straight-line thread programs every instruction
// executes, so the duplicate init always errs in simulation first and the
// static check is not directly observable.)
//
// RUN: not weft standard %s 2>&1 | FileCheck %s
// CHECK: weft: mbarrier 0 initialized twice
// CHECK: weft: failed to simulate the thread programs
module {
  func.func @double_init(%n : index) attributes { "num-threads" = 2 : i64 } {
    // BUG: not guarded to a single thread, so both threads initialize.
    barrier.mbarrier_init 0, 2
    barrier.mbarrier_arrive 0
    return
  }
}
