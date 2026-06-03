module {
  func.func @named_barriers() {
    barrier.named_barrier_arrive 0, 32
    barrier.named_barrier_sync 0, 32
    return
  }
}
