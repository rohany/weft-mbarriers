//===- BarrierPasses.td - Barrier dialect passes ---------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef BARRIER_PASS
#define BARRIER_PASS

include "mlir/Pass/PassBase.td"

def BarrierThreadSpecializePass : Pass<"barrier-thread-specialize", "ModuleOp"> {
  let summary = "Specialize barrier-related functions by thread ID (placeholder).";
  let description = [{
    Empty scaffold; hook for rewriting `barrier.get_tid` and related IR per
    `num-threads` / thread specialization policy. Runs on the top-level `builtin.module`.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

#endif // BARRIER_PASS
