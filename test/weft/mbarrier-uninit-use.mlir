// Arriving on an mbarrier that no thread ever initializes is an error
// (MB-Arrive-Err): the simulation fails on the first use.
//
// RUN: not weft standard %s 2>&1 | FileCheck %s
// CHECK: weft: arrive on uninitialized mbarrier 0
// CHECK: weft: failed to simulate the thread programs
module {
  func.func @uninit_use(%n : index) attributes { "num-threads" = 2 : i64 } {
    barrier.mbarrier_arrive 0
    return
  }
}
