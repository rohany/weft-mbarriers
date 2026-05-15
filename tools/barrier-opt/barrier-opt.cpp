//===- barrier-opt.cpp - Repo-local mlir-opt driver ------------*- C++ -*-===//
//
// Same role as mlir-opt upstream: parses MLIR text and runs passes, but ships
// with the Barrier dialect registered for this repository by default.
//
//===----------------------------------------------------------------------===//

#include "barrier/Dialect/BarrierDialect.h"
#include "barrier/Transform/BarrierPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::barrier::registerPasses();

  mlir::DialectRegistry registry;
  // Core IR dialects (arith, scf, func, cf, memref, tensor, affine, ...).
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::barrier::BarrierDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR Barrier hacking — mlir-opt companion\n", registry));
}
