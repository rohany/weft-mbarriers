//===- BarrierTypes.td - Barrier dialect types -----------*- tablegen -*-===//
//
// Type definitions for the `barrier` dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BARRIER_TYPES
#define BARRIER_TYPES

include "mlir/IR/AttrTypeBase.td"
include "BarrierDialect.td"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// The dialect currently defines no types. Mbarriers are referenced by fixed
// integer ids carried as attributes on the mbarrier operations, mirroring the
// named-barrier operations.

#endif // BARRIER_TYPES
