//===- BarrierFullUnrollLoops.cpp - fully unroll scf.for ---------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "barrier/Transform/BarrierPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::barrier {

#define GEN_PASS_DEF_FULLUNROLLLOOPSPASS
#include "barrier/Transform/BarrierPasses.h.inc"

namespace {

struct FullUnrollLoopsPass final
    : public impl::FullUnrollLoopsPassBase<FullUnrollLoopsPass> {
  using FullUnrollLoopsPassBase::FullUnrollLoopsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect in post-order so that inner loops are unrolled before the outer
    // loops that contain them; this keeps the collected handles valid (an outer
    // unroll would otherwise clone and invalidate the inner `scf.for`s).
    llvm::SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (scf::ForOp forOp : forOps) {
      if (failed(loopUnrollFull(forOp))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::barrier
