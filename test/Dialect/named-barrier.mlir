// RUN: barrier-opt %s | barrier-opt | FileCheck %s

module {
  // CHECK-LABEL: func.func @named_barriers()
  func.func @named_barriers() {
    // CHECK: barrier.named_barrier_arrive 0, 32
    barrier.named_barrier_arrive 0, 32
    // CHECK: barrier.named_barrier_sync 0, 32
    barrier.named_barrier_sync 0, 32
    return
  }
}
