set(LIBS
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRBarrier)

add_llvm_executable(barrier-opt barrier-opt.cpp)

llvm_update_compile_flags(barrier-opt)
target_link_libraries(barrier-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(barrier-opt)
