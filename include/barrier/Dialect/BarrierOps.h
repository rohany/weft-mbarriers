//===- BarrierOps.h - Barrier dialect ops ---------------------*- C++ -*-===//
//
// Operation declarations for the Barrier dialect TableGen emitted headers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BARRIER_BARRIEROPS_H_
#define MLIR_BARRIER_BARRIEROPS_H_

#include "barrier/Dialect/BarrierDialect.h"
#include "barrier/Dialect/BarrierTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "barrier/Dialect/BarrierOps.h.inc"

#endif // MLIR_BARRIER_BARRIEROPS_H_
