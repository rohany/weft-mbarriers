//===- BarrierOps.cpp - Barrier dialect ops --------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "barrier/Dialect/BarrierDialect.h"
#include "barrier/Dialect/BarrierOps.h"

#include "mlir/IR/BuiltinAttributes.h"

#define GET_OP_CLASSES
#include "barrier/Dialect/BarrierOps.cpp.inc"

using namespace mlir;
using namespace mlir::barrier;
