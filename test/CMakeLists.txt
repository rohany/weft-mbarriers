configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# FileCheck/not/count are imported from the LLVM install (found via the lit
# config's tool search path), so only weft needs to be built here.
set(WEFT_TEST_DEPENDS weft)

add_lit_testsuite(check-weft "Running weft regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${WEFT_TEST_DEPENDS}
)
set_target_properties(check-weft PROPERTIES FOLDER "Tests")

add_lit_testsuites(WEFT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${WEFT_TEST_DEPENDS})

# Register the lit suite with CTest so `make test` / `ctest` runs it from the
# build directory. add_lit_testsuite only creates the custom `check-weft`
# target, which CTest does not pick up on its own.
if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_EXTERNAL_LIT NAMES llvm-lit lit)
endif()

add_test(
  NAME check-weft
  COMMAND "${LLVM_EXTERNAL_LIT}" -v "${CMAKE_CURRENT_BINARY_DIR}"
)
