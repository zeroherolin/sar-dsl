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

# The Python suite covers the frontend, the backend drivers and the emitted
# designs, none of which lit reaches. Register it so `check-sar` is the whole
# story and a lit-only run cannot be mistaken for one.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_custom_target(check-sar-python
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_SOURCE_DIR}/examples
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test/python -q
    DEPENDS ${SAR_DSL_TEST_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running the SAR-DSL Python tests"
    USES_TERMINAL)
  set_target_properties(check-sar-python PROPERTIES FOLDER "Tests")

  add_custom_target(check-sar DEPENDS check-sar-lit check-sar-python)
  set_target_properties(check-sar PROPERTIES FOLDER "Tests")
endif()
