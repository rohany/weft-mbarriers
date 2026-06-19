// Two-CTA Jacobi smoother cluster sync from
// https://metaworld.me/blog/public/Statically-finding-races-in-CUTE-kernels-or-Proving-absences-of-Deadlocks
//
// But, this program has the bug described in the blog post, where if phase
// is not updated, it works for n=1 but not higher than that.
//
// Modeled as 2 CTAs x 2 threads (4 total); expected arrivals per gate = 4.
//
// RUN: weft standard %s -n 1 | FileCheck %s --check-prefix=K1
// RUN: not weft standard %s -n 2 2>&1 | FileCheck %s --check-prefix=K2
// K1: weft: the program is race-free
// K2: weft: failed to simulate the thread program
module {
  func.func @jacobi_mbarrier_phase(%n : index) attributes { "num-threads" = 4 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1
    %is_cta0 = arith.cmpi ult, %tid, %c2_i32 : i32
    %local_tid = arith.remui %tid, %c2_i32 : i32
    %is_local_tid0 = arith.cmpi eq, %local_tid, %c0_i32 : i32
    %gate = barrier.mbarrier_new 0, 4
    scf.if %is_cta0 {
      scf.for %i = %c0 to %n step %c1 iter_args(%phase = %false) -> (i1) {
        // tid 0 publishes its edge cell into the peer halo slot over cluster DSMEM.
        scf.if %is_local_tid0 {
          barrier.smem_write 0
        }
        barrier.mbarrier_arrive %gate
        barrier.mbarrier_wait %gate %false
        %phase_next = arith.xori %phase, %true : i1
        scf.yield %phase_next : i1
      }
    } else {
      scf.for %j = %c0 to %n step %c1 iter_args(%phase = %false) -> (i1) {
        scf.if %is_local_tid0 {
          barrier.smem_write 1
        }
        barrier.mbarrier_arrive %gate
        barrier.mbarrier_wait %gate %false
        %phase_next = arith.xori %phase, %true : i1
        scf.yield %phase_next : i1
      }
    }
    return
  }
}