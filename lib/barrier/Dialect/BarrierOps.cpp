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

LogicalResult NewOp::verify() {
  auto attr = getOperation()->getAttr(kBarrierIdAttrName);
  if (!attr)
    return emitOpError() << "requires a '" << kBarrierIdAttrName
                         << "' i64 attribute";
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr || intAttr.getType() != IntegerType::get(getContext(), 64))
    return emitOpError() << "'" << kBarrierIdAttrName
                         << "' must be a 64-bit integer attribute";
  return success();
}

int64_t NewOp::getBarrierId() {
  return cast<IntegerAttr>(getOperation()->getAttr(kBarrierIdAttrName)).getInt();
}
