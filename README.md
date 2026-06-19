# Build Instructions

```
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

If `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is not provided, then VSCode's clangd becomes unhappy and won't
properly highlight or typecheck MLIR/LLVM types and includes.

# Example

After building, an example of the check can be run with:
```
./bin/weft standard ../test/weft/prod-cons.mlir -n 2
```

Which unrolls the loop in prod-cons.mlir by 2 and then applies the Weft race-detection algorithm.

# Testing

Run either

```
make && make test
```

or 

```
make && ctest -V
```

within the build directory to run tests. The latter command gives much more verbose output.
