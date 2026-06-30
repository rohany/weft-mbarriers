// Four threads each arrive once on named barrier 0, but the barrier is
// configured with expected arrival count 2. Only two threads can satisfy each
// phase, so generation 1 can begin before all four threads have finished
// their first arrive -- weft should reject this as not well-synchronized.
//
// RUN: not weft standard %s -n 1 2>&1 | FileCheck %s
// CHECK: weft: the program is not well-synchronized
module {
  func.func @named_barrier_arrive_count_mismatch(%n : index)
      attributes { "num-threads" = 4 : i64 } {
    barrier.named_barrier_arrive 0, 2
    return
  }
}
