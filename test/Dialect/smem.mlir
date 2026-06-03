module {
  func.func @smem() {
    // Side-effecting reads/writes to compile-time shared-memory addresses.
    barrier.smem_write 1024
    barrier.smem_read 1024
    barrier.smem_write 2048
    barrier.smem_read 2048
    return
  }
}
