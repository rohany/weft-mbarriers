// RUN: weft %s -n 4 | FileCheck %s
// CHECK: weft: the program is race-free
module {
  func.func @prod_cons(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant 1 : i1
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    // The "full" signal (producer -> consumer) is carried by an mbarrier, while
    // the "empty" signal (consumer -> producer) is carried by named barrier 0.
    %b1 = barrier.mbarrier_new 1, 1
    scf.if %is_tid0 {
      scf.for %i = %c0 to %n step %c1 {
        // Wait for the buffer to be empty via named barrier 0.
        barrier.named_barrier_sync 0, 2
        barrier.smem_write 0
        // Signal that the buffer is full via mbarrier b1.
        barrier.mbarrier_arrive %b1
      }
    } else {
      // Do an initial arrive to kick off the pipeline (buffer starts empty).
      barrier.named_barrier_arrive 0, 2
      %p1 = arith.constant 0 : i1
      %for1 = scf.for %j = %c0 to %n step %c1 iter_args(%a1 = %p1) -> (i1) {
        // Wait for the buffer to be full via mbarrier b1.
        barrier.mbarrier_wait %b1 %a1
        barrier.smem_read 0
        // Signal that the buffer is empty via named barrier 0.
        barrier.named_barrier_arrive 0, 2
        %xor = arith.xori %a1, %true : i1
        scf.yield %xor : i1
      }
    }
    return
  }
}
