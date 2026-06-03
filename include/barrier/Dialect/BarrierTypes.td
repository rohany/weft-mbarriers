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

def Barrier_MbarrierType : TypeDef<Barrier_Dialect, "Mbarrier"> {
  let mnemonic = "mbarrier";
  let summary = "PTX mbarrier synchronization type.";
  let description = [{
    Represents a PTX mbarrier used for synchronization experiments in this
    dialect. Distinct from future named-barrier types.
  }];
}

#endif // BARRIER_TYPES
