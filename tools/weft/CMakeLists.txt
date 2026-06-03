set(LIBS
    MLIRBarrier
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions)

add_llvm_executable(weft weft.cpp)

llvm_update_compile_flags(weft)
target_link_libraries(weft PRIVATE ${LIBS})
mlir_check_all_link_libraries(weft)
