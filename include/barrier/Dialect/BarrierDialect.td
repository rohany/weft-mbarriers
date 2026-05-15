//===- BarrierDialect.td - Barrier dialect -----------------*- tablegen -*-===//
//
// Stub TableGen definitions for the `barrier` dialect. Extend this file with
// real dialect attributes, traits, interfaces, etc.
//
//===----------------------------------------------------------------------===//

#ifndef BARRIER_DIALECT
#define BARRIER_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Barrier dialect definition.
//===----------------------------------------------------------------------===//

def Barrier_Dialect : Dialect {
  let name = "barrier";
  let summary = "Out-of-tree Barrier dialect stub.";
  let description = [{
    Placeholder dialect for concurrency / synchronization experiments.
    Extend with operations, attributes, types, and passes alongside the matching
    C++ sources under `lib/barrier/Dialect/`.
  }];
  let cppNamespace = "::mlir::barrier";

  // Enable generated parseType/printType dispatch for TableGen TypeDefs.
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Barrier operation placeholder.
//===----------------------------------------------------------------------===//

class Barrier_Op<string mnemonic, list<Trait> traits = []>
    : Op<Barrier_Dialect, mnemonic, traits>;

#endif // BARRIER_DIALECT
