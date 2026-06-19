module {
  func.func @smem() {
    // Side-effecting reads/writes to shared-memory addresses.
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    barrier.smem_write %c1024 : index
    barrier.smem_read %c1024 : index
    barrier.smem_write %c2048 : index
    barrier.smem_read %c2048 : index
    return
  }
}
