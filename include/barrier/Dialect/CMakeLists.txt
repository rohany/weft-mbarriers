add_mlir_dialect(BarrierOps barrier)
add_mlir_doc(BarrierDialect BarrierDialect barrier/Dialect/ -gen-dialect-doc)
add_mlir_doc(BarrierOps BarrierOps barrier/Dialect/ -gen-op-doc)
add_mlir_doc(BarrierTypes BarrierTypes barrier/Dialect/ -gen-typedef-doc
             --typedefs-dialect=barrier)

set(LLVM_TARGET_DEFINITIONS BarrierTypes.td)
mlir_tablegen(BarrierTypes.h.inc --gen-typedef-decls --typedefs-dialect=barrier)
mlir_tablegen(BarrierTypes.cpp.inc --gen-typedef-defs --typedefs-dialect=barrier)
add_mlir_dialect_tablegen_target(MLIRBarrierTypesIncGen)
