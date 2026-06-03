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

def Barrier_MbarrierNewOp : Barrier_Op<"mbarrier_new", []> {
  let summary = "Create a new mbarrier";
  let description = [{
    Create and return a new PTX mbarrier instance. The required i64
    `barrier_id` attribute distinguishes logical mbarrier identities (e.g.
    across thread specialization or analyses). The required i64
    `arrival_count` attribute is the number of arrivals required before the
    barrier completes a phase.
  }];

  let arguments = (ins I64Attr:$barrier_id, I64Attr:$arrival_count);
  let results = (outs Barrier_MbarrierType:$mbarrier);

  let assemblyFormat = "$barrier_id `,` $arrival_count attr-dict";
}

def Barrier_MbarrierArriveOp : Barrier_Op<"mbarrier_arrive", []> {
  let summary = "Mbarrier arrival";
  let description = "Arrive on a PTX mbarrier.";

  let arguments = (ins Barrier_MbarrierType:$mbarrier);
  let assemblyFormat = "$mbarrier attr-dict";
}

def Barrier_MbarrierWaitOp : Barrier_Op<"mbarrier_wait", []> {
  let summary = "Mbarrier wait";
  let description = "Wait on a PTX mbarrier, conditioned on an i1 value.";

  let arguments = (ins Barrier_MbarrierType:$mbarrier, I1:$cond);
  let results = (outs);
  let assemblyFormat = "$mbarrier $cond attr-dict";
}

def Barrier_NamedBarrierArriveOp : Barrier_Op<"named_barrier_arrive", []> {
  let summary = "PTX named barrier arrive";
  let description = [{
    Non-blocking arrival on a PTX named barrier (`bar.arrive`). The required
    i64 `barrier_id` is the named barrier index (0--15). The required i64
    `arrival_count` is the number of participating threads.
  }];

  let arguments = (ins I64Attr:$barrier_id, I64Attr:$arrival_count);
  let results = (outs);

  let assemblyFormat = "$barrier_id `,` $arrival_count attr-dict";
}

def Barrier_NamedBarrierSyncOp : Barrier_Op<"named_barrier_sync", []> {
  let summary = "PTX named barrier sync";
  let description = [{
    Blocking sync on a PTX named barrier (`bar.sync`). The required i64
    `barrier_id` is the named barrier index (0--15). The required i64
    `arrival_count` is the number of participating threads.
  }];

  let arguments = (ins I64Attr:$barrier_id, I64Attr:$arrival_count);
  let results = (outs);

  let assemblyFormat = "$barrier_id `,` $arrival_count attr-dict";
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
