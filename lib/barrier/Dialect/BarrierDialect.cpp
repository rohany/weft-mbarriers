//===- BarrierDialect.cpp - Barrier dialect --------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "barrier/Dialect/BarrierDialect.h"
#include "barrier/Dialect/BarrierOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::barrier;

#include "barrier/Dialect/BarrierOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Barrier dialect.
//===----------------------------------------------------------------------===//

void BarrierDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "barrier/Dialect/BarrierOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "barrier/Dialect/BarrierTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "barrier/Dialect/BarrierTypes.cpp.inc"
