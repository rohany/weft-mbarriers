Use (or create) a build in the `build/` directory. Do not build in the root of the project and pollute the repository with cmake build files.

Check if weft passes all existing tests with `make test` within the build directory, though you need to be in an environment that has LLVM lit and FileCheck.