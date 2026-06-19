// RUN: barrier-opt %s | barrier-opt | FileCheck %s

module {
  // CHECK-LABEL: func.func @smem()
  func.func @smem() {
    // Side-effecting reads/writes to shared-memory addresses.
    // CHECK: %[[C1024:.*]] = arith.constant 1024 : index
    %c1024 = arith.constant 1024 : index
    // CHECK: %[[C2048:.*]] = arith.constant 2048 : index
    %c2048 = arith.constant 2048 : index
    // CHECK: barrier.smem_write %[[C1024]] : index
    barrier.smem_write %c1024 : index
    // CHECK: barrier.smem_read %[[C1024]] : index
    barrier.smem_read %c1024 : index
    // CHECK: barrier.smem_write %[[C2048]] : index
    barrier.smem_write %c2048 : index
    // CHECK: barrier.smem_read %[[C2048]] : index
    barrier.smem_read %c2048 : index
    return
  }
}
