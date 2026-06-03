import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "WEFT"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.weft_obj_root

# Only the test/weft suite carries weft RUN lines; the test/Dialect inputs are
# out of scope (no num-threads attribute), so keep them out of discovery.
config.excludes = ["Dialect"]

# FileCheck/not/count live in <llvm_install_prefix>/libexec/llvm in this
# environment rather than the bin dir that LLVM_TOOLS_BINARY_DIR points at.
# use_default_substitutions() resolves those tools against config.llvm_tools_dir,
# so point it at the directory that actually contains them.
libexec_llvm = os.path.join(config.llvm_install_prefix, "libexec", "llvm")
if os.path.isdir(libexec_llvm):
    config.llvm_tools_dir = libexec_llvm

# Put the LLVM tools dir on PATH so a leading `not` in a RUN line resolves
# (use_default_substitutions only rewrites the piped `| not` form).
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

llvm_config.use_default_substitutions()

# weft is built into the project's own bin directory.
llvm_config.add_tool_substitutions(["weft"], [config.weft_tools_dir])
