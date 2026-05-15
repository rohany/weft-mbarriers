# Build Instructions

```
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

If `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is not provided, then VSCode's clangd becomes unhappy and won't
properly highlight or typecheck MLIR/LLVM types and includes.
