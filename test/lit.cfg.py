# -*- Python -*-
"""Lit configuration for the SAR-DSL MLIR tests."""

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "SAR-DSL"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.sar_dsl_obj_root, "test")
config.excludes = ["CMakeLists.txt", "python"]

llvm_config.use_default_substitutions()

tool_dirs = [config.sar_dsl_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    ["sar-opt", "sar-translate", "mlir-opt", "FileCheck"], tool_dirs)
