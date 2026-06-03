module {
  func.func @prod_cons(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      scf.for %i = %c0 to %n step %c1 {
        barrier.named_barrier_sync 0, 2
        barrier.smem_write 0
        barrier.named_barrier_arrive 1, 2
      }
    } else {
      // Do an initial arrive to kick off the pipeline.
      barrier.named_barrier_arrive 0, 2
      scf.for %j = %c0 to %n step %c1 {
        barrier.named_barrier_sync 1, 2
        barrier.smem_read 0
        barrier.named_barrier_arrive 0, 2
      }
    }
    return
  }
}
