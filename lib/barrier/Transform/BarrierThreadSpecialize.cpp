//===- BarrierThreadSpecialize.cpp - barrier-thread-specialize -*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "barrier/Dialect/BarrierDialect.h"
#include "barrier/Dialect/BarrierOps.h"
#include "barrier/Transform/BarrierPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

namespace mlir::barrier {

#define GEN_PASS_DEF_BARRIERTHREADSPECIALIZEPASS
#include "barrier/Transform/BarrierPasses.h.inc"

namespace {

void specialize_function(ModuleOp parent, func::FuncOp func, int tid) {
  // Clone the function and make a new name.
  auto cloned = cast<func::FuncOp>(func->clone());
  std::string newName =
      (llvm::Twine(func.getSymName()) + "_" + llvm::Twine(tid)).str();
  cloned.setSymName(newName);
  parent.getBody()->push_back(cloned);

  // Add an attribute to track that we specialized this function.
  mlir::MLIRContext *ctx = parent.getContext();
  cloned->removeAttr(kNumThreadsAttrName);
  cloned->setAttr(kThreadSpecializedAttrName,
                  IntegerAttr::get(IntegerType::get(ctx, 64), tid));

  // Replace all get_tid operations with tid.
  llvm::SmallVector<GetTidOp> getTidOps;
  cloned.walk([&](GetTidOp op) {
    getTidOps.push_back(op);
  });
  for (auto op : getTidOps) {
    mlir::OpBuilder b(op);
    auto constant = arith::ConstantOp::create(b, op.getLoc(), op.getType(), b.getI32IntegerAttr(tid));
    op.replaceAllUsesWith(constant.getResult());
    op.erase();
  }
}

// Recording some notes here about if this is actually how I want to do this. A problem
// with this setup right now is that by splitting things into separate functions, I now have
// two separate groups of operations (i.e. the barriers are now different), and I would have
// to do some kind of alignment procedure to stick the barriers back together for some kind
// of analysis. But, this has still been significantly more successful in constructing something
// about the kinds of programs I want to analyze than staring at the PTX.

struct BarrierThreadSpecializePass final
    : public impl::BarrierThreadSpecializePassBase<BarrierThreadSpecializePass> {
  using impl::BarrierThreadSpecializePassBase<
      BarrierThreadSpecializePass>::BarrierThreadSpecializePassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto funcs = llvm::to_vector(module.getOps<func::FuncOp>());
    if (funcs.size() != 1) {
      module.emitOpError()
          << "requires exactly one function in the module, but found "
          << funcs.size();
      signalPassFailure();
      return;
    }
    func::FuncOp func = funcs.front();
    auto threadsAttr =
        func->getAttrOfType<IntegerAttr>(kNumThreadsAttrName);
    if (!threadsAttr) {
      func.emitOpError()
          << "requires \"" << kNumThreadsAttrName
          << "\" as a builtin integer attribute (e.g. `2 : i64`)";
      signalPassFailure();
      return;
    }
    int64_t numThreads = threadsAttr.getInt();
    for (int tid = 0; tid < numThreads; ++tid) {
      specialize_function(module, func, tid);
    }
    func.erase();
  }
};

} // namespace

} // namespace mlir::barrier
