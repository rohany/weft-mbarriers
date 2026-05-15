//===- BarrierPasses.h - Barrier passes -----------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BARRIER_TRANSFORM_BARRIERPASSES_H_
#define MLIR_BARRIER_TRANSFORM_BARRIERPASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::barrier {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "barrier/Transform/BarrierPasses.h.inc"

} // namespace mlir::barrier

#endif // MLIR_BARRIER_TRANSFORM_BARRIERPASSES_H_
