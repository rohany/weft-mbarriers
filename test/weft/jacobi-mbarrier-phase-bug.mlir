// Jacobi stencil smoother with mbarrier phase tracking -- BUGGY variant.
//
// This is jacobi-mbarrier-phase.mlir with the bug described in the blog post:
// the per-iteration phase bit is never fed to the waits. Both waits always
// pass `%false`, so the program works for n=1 (the first phase really is 0)
// but deadlocks for n>1, because after the first completion each gate's phase
// has flipped to 1 and no thread ever waits on it.
//
// Every thread owns the shared-memory cell at its global thread id and
// computes a 3-point stencil over its neighbors: thread t reads cells
// t-1, t, t+1 (skipping absent neighbors at the boundaries) and writes the
// result back into cell t.
//
// Modeled as 4 threads; expected arrivals per gate = 4.
//
// RUN: weft standard %s -n 1 | FileCheck %s --check-prefix=K1
// RUN: not weft standard %s -n 2 2>&1 | FileCheck %s --check-prefix=K2
// K1: weft: the program is race-free
// K2: weft: failed to simulate the thread programs
module {
  func.func @jacobi_mbarrier_phase(%n : index) attributes { "num-threads" = 4 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1

    // Shared-memory addresses for this thread's stencil, derived from its id.
    %left = arith.subi %tid, %c1_i32 : i32
    %right = arith.addi %tid, %c1_i32 : i32

    // Boundary threads (0 and num_threads-1) have no left/right neighbor.
    %has_left = arith.cmpi ne, %tid, %c0_i32 : i32
    %has_right = arith.cmpi ne, %tid, %c3_i32 : i32

    // read_gate (mbarrier 0) orders neighbor reads before local writes;
    // write_gate (mbarrier 1) orders this iteration's writes before the next
    // iteration's reads.
    // A single thread initializes the gates; the named-barrier sync
    // (syncthreads) orders the initialization before any thread's use.
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      barrier.mbarrier_init 0, 4
      barrier.mbarrier_init 1, 4
    }
    barrier.named_barrier_sync 0, 4

    scf.for %i = %c0 to %n step %c1 iter_args(%phase = %false) -> (i1) {
      // Stencil read: thread t reads cells t-1, t, t+1, skipping any
      // neighbor it doesn't have at the boundaries.
      scf.if %has_left {
        barrier.smem_read %left : i32
      }
      barrier.smem_read %tid : i32
      scf.if %has_right {
        barrier.smem_read %right : i32
      }

      // Ensure every thread has read its neighbors before anyone overwrites
      // its own cell.
      barrier.mbarrier_arrive 0
      // BUG: should wait on %phase, not the constant %false.
      barrier.mbarrier_wait 0 %false

      // Stencil write: thread t writes the accumulated result into cell t.
      barrier.smem_write %tid : i32

      // Ensure this iteration's writes are done before the next iteration's
      // reads observe the neighboring cells.
      barrier.mbarrier_arrive 1
      // BUG: should wait on %phase, not the constant %false.
      barrier.mbarrier_wait 1 %false

      %phase_next = arith.xori %phase, %true : i1
      scf.yield %phase_next : i1
    }
    return
  }
}
