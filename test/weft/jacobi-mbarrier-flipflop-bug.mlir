// Jacobi stencil smoother with double-buffered (flip-flop) shared memory --
// BUGGY variant.
//
// This is jacobi-mbarrier-flipflop.mlir with the bug described in the blog
// post: the per-iteration phase bit is never fed to the wait. The wait always
// passes `%false`, so the program works for n=1 (the first phase really is 0)
// but deadlocks for n>1, because after the first completion the gate's phase
// has flipped to 1 and no thread ever waits on it.
//
// Every thread owns one cell and computes a 3-point stencil over its
// neighbors, reading cells t-1, t, t+1 from the *current* buffer (offset 0 or
// 4) and writing cell t into the *next* buffer. The offsets swap each
// iteration.
//
// Modeled as 4 threads; expected arrivals per gate = 4.
//
// RUN: weft standard %s -n 1 | FileCheck %s --check-prefix=K1
// RUN: not weft standard %s -n 2 2>&1 | FileCheck %s --check-prefix=K2
// K1: weft: the program is race-free
// K2: weft: failed to simulate the thread programs
module {
  func.func @jacobi_mbarrier_flipflop(%n : index) attributes { "num-threads" = 4 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1

    // Per-thread stencil offsets, derived from the thread id.
    %left = arith.subi %tid, %c1_i32 : i32
    %right = arith.addi %tid, %c1_i32 : i32

    // Boundary threads (0 and num_threads-1) have no left/right neighbor.
    %has_left = arith.cmpi ne, %tid, %c0_i32 : i32
    %has_right = arith.cmpi ne, %tid, %c3_i32 : i32

    // A single gate orders each iteration's writes before the next
    // iteration's reads; the disjoint buffers handle the anti-dependency.
    %gate = barrier.mbarrier_new 0, 4

    // Loop-carried state: current buffer offset, next buffer offset, phase.
    // The two offsets (0 and 4) swap every iteration.
    scf.for %i = %c0 to %n step %c1
        iter_args(%cur = %c0_i32, %next = %c4_i32, %phase = %false) -> (i32, i32, i1) {
      // Absolute stencil addresses in the current buffer.
      %cur_left = arith.addi %cur, %left : i32
      %cur_center = arith.addi %cur, %tid : i32
      %cur_right = arith.addi %cur, %right : i32

      // Stencil read from the current buffer (skipping absent neighbors).
      scf.if %has_left {
        barrier.smem_read %cur_left : i32
      }
      barrier.smem_read %cur_center : i32
      scf.if %has_right {
        barrier.smem_read %cur_right : i32
      }

      // Stencil write into this thread's cell in the next buffer.
      %next_center = arith.addi %next, %tid : i32
      barrier.smem_write %next_center : i32

      // Single barrier: this iteration's writes complete before the next
      // iteration's reads observe the (now current) buffer.
      barrier.mbarrier_arrive %gate
      // BUG: should wait on %phase, not the constant %false.
      barrier.mbarrier_wait %gate %false

      // Flip-flop the buffers and toggle the phase for the next round.
      %phase_next = arith.xori %phase, %true : i1
      scf.yield %next, %cur, %phase_next : i32, i32, i1
    }
    return
  }
}
