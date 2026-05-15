//===- BarrierDialect.h - Barrier dialect ----------------------*- C++ -*-===//
//
// Public dialect declaration. Expand as you introduce types, bytecode support,
// and additional interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BARRIER_BARRIERDIALECT_H_
#define MLIR_BARRIER_BARRIERDIALECT_H_

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Dialect.h"

#include "barrier/Dialect/BarrierOpsDialect.h.inc"

namespace mlir::barrier {

/// Legacy scheduling attribute copied by `func.func` clone; removed from
/// thread-specialized clones.
inline constexpr llvm::StringLiteral kNumThreadsAttrName = "num-threads";

/// Mark on `func.func` clones produced by thread specialization; value is the
/// thread id used for that clone.
inline constexpr llvm::StringLiteral kThreadSpecializedAttrName =
    "thread-specialized";

} // namespace mlir::barrier

#endif // MLIR_BARRIER_BARRIERDIALECT_H_
