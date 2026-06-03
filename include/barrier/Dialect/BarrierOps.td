//===- BarrierOps.td - Barrier dialect ops -----------------*- tablegen -*-===//
//
// Extend this file with real operations and interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef BARRIER_OPS
#define BARRIER_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "BarrierDialect.td"
include "BarrierTypes.td"

def Barrier_NewOp : Barrier_Op<"new", []> {
  let summary = "Create a new barrier";
  let description = [{
    Operation to create and return a new instance of a barrier.
    A required i64 attribute named "barrier-id" distinguishes logical
    barrier identities (e.g. across thread specialization or analyses).
  }];

  let results = (outs Barrier_BarrierType:$barrier);

  let assemblyFormat = "attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static constexpr llvm::StringLiteral kBarrierIdAttrName =
        "barrier-id";

    int64_t getBarrierId();
  }];
}

def Barrier_ArriveOp : Barrier_Op<"arrive", []> {
  let summary = "Barrier arrival";
  let description = "Barrier arrival";

  let arguments = (ins Barrier_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict";
}

def Barrier_WaitOp : Barrier_Op<"wait", []> {
  let summary = "Barrier wait";
  let description = "Wait on a barrier, conditioned on an i1 value.";

  let arguments = (ins Barrier_BarrierType:$barrier, I1:$cond);
  let results = (outs);
  let assemblyFormat = "$barrier $cond attr-dict";
}

def Barrier_GetTidOp : Barrier_Op<"get_tid", [Pure]> {
  let summary = "Get thread ID";
  let description = [{
    Returns the current thread's ID as a 32-bit integer.

    This operation is pure, has no arguments, and returns an i32.
  }];

  let results = (outs I32:$tid);

  let assemblyFormat = "attr-dict";
}

// Dummy operations that I'm just using to make looking at the code
// I'm writing more clear.

def Barrier_TMALoadOp : Barrier_Op<"tma_load", []> {
  let summary = "Dummy side-effecting TMA load operation";
  let description = [{
    Dummy operation in the barrier dialect representing a side-effecting
    TMA load. Used for testing infrastructure or as a placeholder for
    potential hardware integration.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Barrier_TCMMAOp : Barrier_Op<"tc_mma", []> {
  let summary = "Dummy side-effecting TC MMA operation";
  let description = [{
    Dummy operation in the barrier dialect representing a side-effecting
    TC MMA (matrix multiply-accumulate) operation. Serves as a placeholder
    for more complex or hardware-backed features.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}


#endif // BARRIER_OPS
