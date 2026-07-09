// RUN: barrier-opt %s | barrier-opt | FileCheck %s

module {
  // CHECK-LABEL: func.func @mbarriers()
  func.func @mbarriers() {
    %false = arith.constant 0 : i1
    // CHECK: barrier.mbarrier_init 0, 4
    barrier.mbarrier_init 0, 4
    // CHECK: barrier.mbarrier_arrive 0
    barrier.mbarrier_arrive 0
    // CHECK: barrier.mbarrier_wait 0 %{{.*}}
    barrier.mbarrier_wait 0 %false
    return
  }
}
