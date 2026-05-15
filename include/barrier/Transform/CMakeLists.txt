set(LLVM_TARGET_DEFINITIONS BarrierPasses.td)
mlir_tablegen(BarrierPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRBarrierPassesIncGen)
