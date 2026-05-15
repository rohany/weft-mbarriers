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

def Barrier_BarrierType : TypeDef<Barrier_Dialect, "Barrier"> {
  let mnemonic = "barrier";
  let summary = "Barrier synchronization type.";
  let description = [{
    Represents a barrier used for synchronization experiments in this dialect.
    Parameterize or split into distinct types as semantics evolve.
  }];
}

#endif // BARRIER_TYPES
