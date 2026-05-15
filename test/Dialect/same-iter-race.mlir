module {
  func.func @same_iter_race(%n : index) attributes { "num-threads" = 2 : i64 } {
    %tid = barrier.get_tid
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant 1 : i1
    %is_tid0 = arith.cmpi eq, %tid, %c0_i32 : i32
    %b0 = barrier.new
    %b1 = barrier.new
    scf.if %is_tid0 {
      %p0 = arith.constant 0 : i1
      %for0 = scf.for %i = %c0 to %n step %c1 iter_args(%a0 = %p0) -> (i1) {

        barrier.wait %b0 %a0
        barrier.tma_load
        barrier.arrive %b1

        %xor = arith.xori %a0, %true : i1
        scf.yield %xor : i1
      }
    } else {
      %p1 = arith.constant 0 : i1
      %for1 = scf.for %j = %c0 to %n step %c1 iter_args(%a1 = %p1) -> (i1) {

        barrier.arrive %b0
        barrier.tc_mma
        barrier.wait %b1 %p1

        %xor = arith.xori %a1, %true : i1
        scf.yield %xor : i1
      }
    }
    return
  }
}
