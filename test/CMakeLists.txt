configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SAR_DSL_TEST_DEPENDS
  FileCheck
  sar-opt
  sar-translate
  sar_runtime
)

add_lit_testsuite(check-sar-lit "Running the SAR-DSL lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SAR_DSL_TEST_DEPENDS}
)
set_target_properties(check-sar-lit PROPERTIES FOLDER "Tests")
