// RUN: not weft standard %s -n 4 2>&1 | FileCheck %s
// CHECK: weft: race detected on address 0
// CHECK: weft: the program has a race
module {
  func.func @same_iter_race(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant 1 : i1
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    // A single thread initializes the mbarriers; the named-barrier sync
    // (syncthreads) orders the initialization before any thread's use.
    scf.if %is_tid0 {
      barrier.mbarrier_init 0, 1
      barrier.mbarrier_init 1, 1
    }
    barrier.named_barrier_sync 0, 2
    scf.if %is_tid0 {
      %p0 = arith.constant 0 : i1
      %for0 = scf.for %i = %c0 to %n step %c1 iter_args(%a0 = %p0) -> (i1) {

        barrier.mbarrier_wait 0 %a0
        barrier.smem_write %c0 : index
        barrier.mbarrier_arrive 1

        %xor = arith.xori %a0, %true : i1
        scf.yield %xor : i1
      }
    } else {
      %p1 = arith.constant 0 : i1
      %for1 = scf.for %j = %c0 to %n step %c1 iter_args(%a1 = %p1) -> (i1) {

        barrier.mbarrier_arrive 0
        barrier.smem_read %c0 : index
        barrier.mbarrier_wait 1 %a1

        %xor = arith.xori %a1, %true : i1
        scf.yield %xor : i1
      }
    }
    return
  }
}
