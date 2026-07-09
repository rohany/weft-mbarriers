// Jacobi stencil smoother with double-buffered (flip-flop) shared memory,
// following the pattern from
// https://metaworld.me/blog/public/Statically-finding-races-in-CUTE-kernels-or-Proving-absences-of-Deadlocks
//
// Every thread owns one cell and computes a 3-point stencil over its
// neighbors: thread t reads cells t-1, t, t+1 from the *current* buffer and
// writes the result into cell t of the *next* buffer. The two buffers live at
// shared-memory offsets 0 (current) and 4 (next); the offsets swap every
// iteration so the roles flip-flop.
//
// Unlike jacobi-mbarrier-phase.mlir -- which needs two gates to separate the
// forward dependency (next iteration reads this iteration's writes) from the
// anti-dependency (this iteration's reads must finish before the cell is
// overwritten) -- double buffering removes the anti-dependency entirely:
// reads hit the current buffer while writes hit the next buffer, which are
// disjoint. A single gate per iteration therefore suffices, ordering each
// iteration's writes before the next iteration's reads. The phase toggles per
// iteration (iter 0 waits on 0, iter 1 on 1, iter 2 on 0 again).
//
// Modeled as 4 threads; expected arrivals per gate = 4.
//
// RUN: weft standard %s -n 3 | FileCheck %s
// CHECK: weft: the program is race-free
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

    // A single gate (mbarrier 0) orders each iteration's writes before the
    // next iteration's reads; the disjoint buffers handle the
    // anti-dependency. A single thread initializes the gate; the
    // named-barrier sync (syncthreads) orders the initialization before any
    // thread's use.
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      barrier.mbarrier_init 0, 4
    }
    barrier.named_barrier_sync 0, 4

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
      barrier.mbarrier_arrive 0
      barrier.mbarrier_wait 0 %phase

      // Flip-flop the buffers and toggle the phase for the next round.
      %phase_next = arith.xori %phase, %true : i1
      scf.yield %next, %cur, %phase_next : i32, i32, i1
    }
    return
  }
}
