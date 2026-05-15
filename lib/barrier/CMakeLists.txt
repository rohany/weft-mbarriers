add_mlir_dialect_library(
  MLIRBarrier
  Dialect/BarrierDialect.cpp
  Dialect/BarrierOps.cpp
  Transform/BarrierPasses.cpp
  Transform/BarrierThreadSpecialize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/barrier/Dialect
  ${PROJECT_SOURCE_DIR}/include/barrier/Transform

  DEPENDS
  MLIRBarrierOpsIncGen
  MLIRBarrierTypesIncGen
  MLIRBarrierPassesIncGen

  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
