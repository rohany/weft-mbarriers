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
    Operation to create and return a new instance of a barrier. The required
    i64 `barrier_id` argument distinguishes logical barrier identities (e.g.
    across thread specialization or analyses).
  }];

  let arguments = (ins I64Attr:$barrier_id);
  let results = (outs Barrier_BarrierType:$barrier);

  let assemblyFormat = "$barrier_id attr-dict";
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

def Barrier_SmemReadOp : Barrier_Op<"smem_read", []> {
  let summary = "Read from a compile-time shared-memory address";
  let description = [{
    Side-effecting read from a fixed (compile-time) shared-memory address
    given by the `address` attribute. Returns no data.
  }];

  let arguments = (ins I64Attr:$address);
  let results = (outs);

  let assemblyFormat = "$address attr-dict";
}

def Barrier_SmemWriteOp : Barrier_Op<"smem_write", []> {
  let summary = "Write to a compile-time shared-memory address";
  let description = [{
    Side-effecting write to a fixed (compile-time) shared-memory address
    given by the `address` attribute.
  }];

  let arguments = (ins I64Attr:$address);
  let results = (outs);

  let assemblyFormat = "$address attr-dict";
}


#endif // BARRIER_OPS
