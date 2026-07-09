// Jacobi stencil smoother with a single shared-memory buffer and a single
// barrier per iteration -- BUGGY variant.
//
// Every thread owns the shared-memory cell at its global thread id and
// computes a 3-point stencil over its neighbors: thread t reads cells
// t-1, t, t+1 (skipping absent neighbors at the boundaries) and writes the
// result back into cell t of the *same* buffer.
//
// Unlike jacobi-mbarrier-phase.mlir, this version uses only one barrier per
// iteration (separating the reads from the writes) and unlike
// jacobi-mbarrier-flipflop.mlir it does not double-buffer. The single barrier
// orders this iteration's reads before this iteration's writes, so a single
// iteration (n=1) is race-free. But nothing orders this iteration's write to
// cell t against the *next* iteration's neighbor read of cell t, so for n>1
// the cross-iteration anti-dependency is unsynchronized and races.
//
// Modeled as 4 threads; expected arrivals per gate = 4.
//
// RUN: weft standard %s -n 1 | FileCheck %s --check-prefix=K1
// RUN: not weft standard %s -n 2 2>&1 | FileCheck %s --check-prefix=K2
// K1: weft: the program is race-free
// K2: weft: the program has a race
module {
  func.func @jacobi_mbarrier_single_buffer(%n : index) attributes { "num-threads" = 4 : i64 } {
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

    // A single gate (mbarrier 0) separates this iteration's reads from its
    // writes. It does NOT protect against the next iteration overwriting a
    // cell that this iteration's neighbors still need to read. A single
    // thread initializes the gate; the named-barrier sync (syncthreads)
    // orders the initialization before any thread's use.
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      barrier.mbarrier_init 0, 4
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

      // Single barrier between reads and writes.
      barrier.mbarrier_arrive 0
      barrier.mbarrier_wait 0 %phase

      // Stencil write: thread t writes the result back into its own cell of
      // the single buffer. For n>1 this races the next iteration's reads.
      barrier.smem_write %tid : i32

      %phase_next = arith.xori %phase, %true : i1
      scf.yield %phase_next : i1
    }
    return
  }
}
