// At k=1 the program is well-synchronized: each phase of named barrier 0 sees
// exactly one arrive (from P) and one sync (from Q), satisfying its count of 2.
// At k=2 P can race ahead and arrive twice, satisfying phase 0 by itself; Q's
// sync then blocks forever, which weft detects as a deadlock.
//
// RUN: weft standard %s -n 1 | FileCheck %s --check-prefix=K1
// RUN: not weft standard %s -n 2 2>&1 | FileCheck %s --check-prefix=K2
// RUN: not weft single-nested-loop %s 2>&1 | FileCheck %s --check-prefix=K3
// K1: weft: the program is race-free
// K2: weft: simulation of thread programs has failed due to a deadlock
// K3: weft: simulation of thread programs has failed due to a deadlock
// K3: weft: failed to check well synchronized for 2
module {
  func.func @well_sync_k1_fail_k2(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    scf.if %is_tid0 {
      // P
      scf.for %i = %c0 to %n step %c1 {
        barrier.named_barrier_arrive 0, 2
      }
    } else {
      // Q
      scf.for %j = %c0 to %n step %c1 {
        barrier.named_barrier_sync 0, 2
      }
    }
    return
  }
}
