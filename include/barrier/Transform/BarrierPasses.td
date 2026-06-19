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

def FullUnrollLoopsPass : Pass<"full-unroll-loops", "ModuleOp"> {
  let summary = "Fully unroll every `scf.for` in the module.";
  let description = [{
    Walks the module in post-order (innermost loops first) and applies upstream
    `scf::loopUnrollFull` to each `scf.for`, completely unrolling loops with a
    statically known trip count. A loop that upstream cannot fully unroll (e.g.
    non-constant bounds or step) causes the pass to fail. Runs on
    `builtin.module`.
  }];
  let dependentDialects = ["scf::SCFDialect", "func::FuncDialect"];
}

#endif // BARRIER_PASS
