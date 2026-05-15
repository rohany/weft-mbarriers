//===- BarrierTypes.h - Barrier dialect types ----------------*- C++ -*-===//
//
// TableGen-backed type declarations for the `barrier` dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BARRIER_BARRIERTYPES_H_
#define MLIR_BARRIER_BARRIERTYPES_H_

#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "barrier/Dialect/BarrierTypes.h.inc"

#endif // MLIR_BARRIER_BARRIERTYPES_H_
