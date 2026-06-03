// RUN: weft %s -n 4 | FileCheck %s
// CHECK: weft: the program is race-free
module {
  func.func @prod_cons_pipeline(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    %pb0 = barrier.mbarrier_new 0, 1
    %pb1 = barrier.mbarrier_new 1, 1
    %pb2 = barrier.mbarrier_new 2, 1
    %cb0 = barrier.mbarrier_new 3, 1
    %cb1 = barrier.mbarrier_new 4, 1
    %cb2 = barrier.mbarrier_new 5, 1
    scf.if %is_tid0 {
      // Mimic staged TMA load pipeline: phases on producer slots start high.
      scf.for %i = %c0 to %n step %c1
          iter_args(%ph0 = %true, %ph1 = %true, %ph2 = %true)
          -> (i1, i1, i1) {
        %iu = arith.index_castui %i : index to i32
        %lane = arith.remui %iu, %c3_i32 : i32
        %eq0 = arith.cmpi eq, %lane, %c0_i32 : i32
        %eq1 = arith.cmpi eq, %lane, %c1_i32 : i32
        %next:3 = scf.if %eq0 -> (i1, i1, i1) {
          barrier.mbarrier_wait %pb0 %ph0
          barrier.smem_write 0
          barrier.mbarrier_arrive %cb0
          %ph0_flip = arith.xori %ph0, %true : i1
          scf.yield %ph0_flip, %ph1, %ph2 : i1, i1, i1
        } else {
          %mid:3 = scf.if %eq1 -> (i1, i1, i1) {
            barrier.mbarrier_wait %pb1 %ph1
            barrier.smem_write 1
            barrier.mbarrier_arrive %cb1
            %ph1_flip = arith.xori %ph1, %true : i1
            scf.yield %ph0, %ph1_flip, %ph2 : i1, i1, i1
          } else {
            barrier.mbarrier_wait %pb2 %ph2
            barrier.smem_write 2
            barrier.mbarrier_arrive %cb2
            %ph2_flip = arith.xori %ph2, %true : i1
            scf.yield %ph0, %ph1, %ph2_flip : i1, i1, i1
          }
          scf.yield %mid#0, %mid#1, %mid#2 : i1, i1, i1
        }
        scf.yield %next#0, %next#1, %next#2 : i1, i1, i1
      }
    } else {
      scf.for %j = %c0 to %n step %c1
          iter_args(%qh0 = %false, %qh1 = %false, %qh2 = %false)
          -> (i1, i1, i1) {
        %ju = arith.index_castui %j : index to i32
        %lane = arith.remui %ju, %c3_i32 : i32
        %eq0 = arith.cmpi eq, %lane, %c0_i32 : i32
        %eq1 = arith.cmpi eq, %lane, %c1_i32 : i32
        %next:3 = scf.if %eq0 -> (i1, i1, i1) {
          barrier.mbarrier_wait %cb0 %qh0
          barrier.smem_read 0
          barrier.mbarrier_arrive %pb0
          %qh0_flip = arith.xori %qh0, %true : i1
          scf.yield %qh0_flip, %qh1, %qh2 : i1, i1, i1
        } else {
          %mid:3 = scf.if %eq1 -> (i1, i1, i1) {
            barrier.mbarrier_wait %cb1 %qh1
            barrier.smem_read 1
            barrier.mbarrier_arrive %pb1
            %qh1_flip = arith.xori %qh1, %true : i1
            scf.yield %qh0, %qh1_flip, %qh2 : i1, i1, i1
          } else {
            barrier.mbarrier_wait %cb2 %qh2
            barrier.smem_read 2
            barrier.mbarrier_arrive %pb2
            %qh2_flip = arith.xori %qh2, %true : i1
            scf.yield %qh0, %qh1, %qh2_flip : i1, i1, i1
          }
          scf.yield %mid#0, %mid#1, %mid#2 : i1, i1, i1
        }
        scf.yield %next#0, %next#1, %next#2 : i1, i1, i1
      }
    }
    return
  }
}
