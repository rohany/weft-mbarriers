//===- BarrierScfLoopUnroll.cpp - unroll scf.for by factor --------*- C++
//-*-===//
//
//===----------------------------------------------------------------------===//

#include "barrier/Transform/BarrierPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::barrier {

#define GEN_PASS_DEF_BARRIERSCFLOOPUNROLLPASS
#include "barrier/Transform/BarrierPasses.h.inc"

namespace {

struct BarrierScfLoopUnrollPass final
    : public impl::BarrierScfLoopUnrollPassBase<BarrierScfLoopUnrollPass> {
  using BarrierScfLoopUnrollPassBase::BarrierScfLoopUnrollPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    uint64_t factor = unrollFactor;
    // `loopUnrollJamByFactor` documents factor 1 as a no-op; keep the same UX.
    if (factor < 2)
      return;

    llvm::SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (scf::ForOp forOp : forOps) {
      FailureOr<UnrolledLoopInfo> unr = loopUnrollByFactor(forOp, factor);
      if (failed(unr)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::barrier
