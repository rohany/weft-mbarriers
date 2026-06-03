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
