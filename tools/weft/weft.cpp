//===- weft.cpp - Barrier thread-specialization driver --------*- C++ -*-===//
//
// Reads an MLIR file and runs barrier race-detection analysis. Unlike
// barrier-opt (which is a general mlir-opt clone), weft hard-codes its
// pipelines and exposes a single tuning knob `n`. Usage:
//
//   weft standard <input.mlir> [-n N] [--print-thread-programs]
//   weft single-nested-loop <input.mlir> [-n N] [--print-thread-programs]
//
//===----------------------------------------------------------------------===//

#include "barrier/Dialect/BarrierDialect.h"
#include "barrier/Dialect/BarrierOps.h"
#include "barrier/Transform/BarrierPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <numeric>
#include <optional>
#include <set>

// Specializes every function in `module` that contains a single
// `index` argument with the compile-time constant `n`, then drops
// the now-unused argument from the function signature.
static mlir::LogicalResult specializeEntryArguments(mlir::ModuleOp mod, int n) {
  if (n < 0)
    return mod.emitOpError()
           << "requires the unroll length 'n' to be positive, but found " << n;
  for (auto func : mod.getOps<mlir::func::FuncOp>()) {
    if (func.getNumArguments() != 1)
      continue;
    mlir::BlockArgument arg = func.getArgument(0);
    if (!mlir::isa<mlir::IndexType>(arg.getType()))
      continue;

    // Materialize the constant at the start of the entry block and rewrite
    // every use of the argument to refer to it instead.
    mlir::Block &entry = func.getBody().front();
    mlir::OpBuilder b(&entry, entry.begin());
    auto constant = mlir::arith::ConstantOp::create(
        b, func.getLoc(), b.getIndexType(), b.getIndexAttr(n));
    arg.replaceAllUsesWith(constant.getResult());

    // The argument is now dead; remove it from the block and function type.
    bool erased = succeeded(func.eraseArgument(0));
    assert(erased && "failed to erase the function argument");
    (void)erased;
  }

  return mlir::success();
}

llvm::FailureOr<int> getNumThreads(mlir::ModuleOp module) {
  auto funcs = llvm::to_vector(module.getOps<mlir::func::FuncOp>());
  if (funcs.size() != 1)
    return module.emitOpError()
           << "requires exactly one function in the module, but found "
           << funcs.size();
  mlir::func::FuncOp func = funcs.front();
  mlir::IntegerAttr numThreadsAttr =
      llvm::dyn_cast_if_present<mlir::IntegerAttr>(
          func->getAttr(mlir::barrier::kNumThreadsAttrName));
  if (!numThreadsAttr)
    return func.emitOpError()
           << "input function requires \"" << mlir::barrier::kNumThreadsAttrName
           << "\" as an integer attribute (e.g. `2 : i64`)";
  return numThreadsAttr.getInt();
}

llvm::LogicalResult
simulateThreadPrograms(std::vector<mlir::func::FuncOp> &threadPrograms,
                       std::map<mlir::Operation *, int> &generations) {
  // In this function, we'll simulate a possible execution of each thread
  // program. It doesn't matter how we decide to do this, as we'll then prove
  // later that this is the only execution order that matters if the programs
  // are actually well-synchronized.

  int numThreads = threadPrograms.size();
  std::vector<mlir::Operation *> threadIPs(numThreads);
  for (int i = 0; i < numThreads; i++) {
    mlir::Region &body = threadPrograms[i].getBody();
    if (!body.hasOneBlock()) {
      return threadPrograms[i].emitOpError()
             << "simulation requires exactly one block in the body";
    }
    threadIPs[i] = &(*body.front().begin());
  }

  struct MBarrierState {
    int generation;
    int currentArrivalCount;
    int expectedArrivalCount;
    std::vector<int> waiters;
    MBarrierState(int expectedArrivalCount)
        : generation(0), currentArrivalCount(expectedArrivalCount),
          expectedArrivalCount(expectedArrivalCount) {}
  };

  // Register an arrival on `mbarrier`. When the last expected arrival lands,
  // the barrier flips: every waiting thread observes the current generation, is
  // advanced past its wait, and the generation is bumped for the next phase.
  auto arriveMBarrier = [](std::vector<mlir::Operation *> &threadIPs,
                           std::map<mlir::Operation *, int> &generations,
                           std::set<int> &sleepingThreads,
                           MBarrierState &barrier) {
    barrier.currentArrivalCount--;
    if (barrier.currentArrivalCount == 0) {
      // Advance all waiters registered on this barrier.
      for (auto waiter : barrier.waiters) {
        auto op = threadIPs[waiter];
        // The waiters have observed the barrier at the generation they
        // waited on.
        generations[op] = barrier.generation;
        // The waiter can now proceed.
        threadIPs[waiter] = op->getNextNode();
        // The waiter has been woken, so it is no longer sleeping.
        sleepingThreads.erase(waiter);
      }
      barrier.waiters.clear();
      barrier.generation++;
      barrier.currentArrivalCount = barrier.expectedArrivalCount;
    }
  };

  // MBarrier ID's to current barrier states. A barrier enters the map only
  // when a thread executes its `mbarrier_init` (MB-Init); initializing a
  // barrier twice (MB-Init-Err) or arriving/waiting on a barrier that has
  // not been initialized (MB-Arrive-Err, MB-Wait-Err) is an error.
  std::map<int, MBarrierState> mbarrierStates;

  // Similarly set up state for named barriers.
  struct NamedBarrierState {
    // The generation will be tracked implicitly, but is not
    // exposed to operations. When a generation bumps, we'll reset
    // the current and expected arrival counts to -1, which will
    // signal that the barrier has not yet been "configured" at
    // the current generation.
    int generation;
    int currentArrivalCount;
    int expectedArrivalCount;
    std::vector<int> waiters;
    NamedBarrierState()
        : generation(0), currentArrivalCount(-1), expectedArrivalCount(-1) {}
  };
  // We'll update the barrier states as the program runs, because there
  // isn't an explicit "initialization" step for named barriers.
  std::map<int, NamedBarrierState> namedBarrierStates;
  auto arriveNamedBarrier = [](std::vector<mlir::Operation *> &threadIPs,
                               std::map<mlir::Operation *, int> &generations,
                               std::set<int> &sleepingThreads,
                               NamedBarrierState &barrier,
                               mlir::barrier::NamedBarrierArriveOp op) {
    // If we're the first arrival on this barrier, then we get to
    // configure the barrier.
    int expectedArrivalCount = op.getArrivalCount();
    if (barrier.expectedArrivalCount == -1) {
      barrier.expectedArrivalCount = expectedArrivalCount;
      barrier.currentArrivalCount = expectedArrivalCount;
    } else {
      // Otherwise, we're arriving on a barrier that another thread has
      // already configured. In this case, it's an error that the
      // generations have been configured differently.
      if (barrier.expectedArrivalCount != expectedArrivalCount) {
        llvm::errs() << "weft: named barrier configuration on arrive\n";
        return llvm::failure();
      }
    }
    barrier.currentArrivalCount--;

    // Register this generation as observed.
    generations[op] = barrier.generation;

    // Wake up all the waiters if the barrier is ready.
    if (barrier.currentArrivalCount == 0) {
      for (auto waiter : barrier.waiters) {
        auto op = threadIPs[waiter];
        // The waiters have observed the barrier at the generation they
        // waited on.
        generations[op] = barrier.generation;
        // The waiter can now proceed.
        threadIPs[waiter] = op->getNextNode();
        // The waiter has been woken, so it is no longer sleeping.
        sleepingThreads.erase(waiter);
      }
      barrier.waiters.clear();

      barrier.generation++;
      barrier.currentArrivalCount = -1;
      barrier.expectedArrivalCount = -1;
    }

    return llvm::success();
  };

  auto syncNamedBarrier =
      [](int thread, std::vector<mlir::Operation *> &threadIPs,
         std::map<mlir::Operation *, int> &generations,
         std::set<int> &sleepingThreads, NamedBarrierState &barrier,
         mlir::barrier::NamedBarrierSyncOp op) -> llvm::FailureOr<bool> {
    // We have to do something similar to arrive here first, where we have
    // to check that the configuration on the current generation matches.
    int expectedArrivalCount = op.getArrivalCount();
    if (barrier.expectedArrivalCount == -1) {
      barrier.expectedArrivalCount = expectedArrivalCount;
      barrier.currentArrivalCount = expectedArrivalCount;
    } else {
      if (barrier.expectedArrivalCount != expectedArrivalCount) {
        llvm::errs() << "weft: named barrier configuration on sync\n";
        return llvm::failure();
      }
    }
    barrier.currentArrivalCount--;

    // Register this generation as observed.
    generations[op] = barrier.generation;

    // Wake up all the waiters if the barrier is ready.
    if (barrier.currentArrivalCount == 0) {
      for (auto waiter : barrier.waiters) {
        auto op = threadIPs[waiter];
        // The waiters have observed the barrier at the generation they
        // waited on.
        generations[op] = barrier.generation;
        // The waiter can now proceed.
        threadIPs[waiter] = op->getNextNode();
        // The waiter has been woken, so it is no longer sleeping.
        sleepingThreads.erase(waiter);
      }
      barrier.waiters.clear();

      barrier.generation++;
      barrier.currentArrivalCount = -1;
      barrier.expectedArrivalCount = -1;
      return llvm::FailureOr<bool>(true);
    } else {
      // Otherwise, the caller will fall asleep and register itself as a
      // waiter.
      barrier.waiters.push_back(thread);
      sleepingThreads.insert(thread);
      // The sync operation was a success, but this thread did not take a
      // step.
      return llvm::FailureOr<bool>(false);
    }
  };

  auto areThreadsComplete = [&]() {
    for (int i = 0; i < numThreads; i++) {
      if (threadIPs[i] != threadPrograms[i].getBody().front().getTerminator()) {
        return false;
      }
    }
    return true;
  };

  // Threads that are currently blocked waiting on a barrier. They are skipped
  // when searching for a thread to execute and are removed when the barrier
  // they wait on wakes its waiters.
  std::set<int> sleepingThreads;

  while (true) {
    // Check first if all threads have exited.
    if (areThreadsComplete()) {
      break;
    }

    bool steppedThread = false;
    llvm::LogicalResult stepResult = llvm::success();
    // Try to step a thread. We'll proceed in round-robin order.
    for (int i = 0; i < numThreads; i++) {
      // Skip threads that are asleep on a barrier; only an arrival on that
      // barrier can wake them.
      if (sleepingThreads.count(i))
        continue;

      mlir::Operation *op = threadIPs[i];

      llvm::TypeSwitch<mlir::Operation *, void>(op)
          .Case<mlir::func::ReturnOp>([&](mlir::func::ReturnOp op) {
            // There's nothing to do in the case that a thread is sitting
            // at a return, which is the terminator of the block/function.
          })
          .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp op) {
            // Arithmetic constants don't "do" anything, but we'll step over
            // them.
            steppedThread = true;
          })
          .Case<mlir::barrier::MbarrierInitOp>(
              [&](mlir::barrier::MbarrierInitOp op) {
                // Initialization installs the barrier's state (MB-Init).
                // Initializing an already-initialized barrier is an error
                // (MB-Init-Err).
                int barrierId = op.getBarrierId();
                if (mbarrierStates.find(barrierId) != mbarrierStates.end()) {
                  llvm::errs()
                      << "weft: mbarrier " << barrierId << " initialized "
                      << "twice\n";
                  stepResult = llvm::failure();
                  return;
                }
                mbarrierStates.insert(
                    {barrierId, MBarrierState(op.getArrivalCount())});
                steppedThread = true;
              })
          .Case<mlir::barrier::MbarrierArriveOp>(
              [&](mlir::barrier::MbarrierArriveOp op) {
                // Arrive operations will read the current generation and
                // advance the generation. Arriving on a barrier that has not
                // been initialized is an error (MB-Arrive-Err).
                int barrierId = op.getBarrierId();
                auto it = mbarrierStates.find(barrierId);
                if (it == mbarrierStates.end()) {
                  llvm::errs() << "weft: arrive on uninitialized mbarrier "
                               << barrierId << "\n";
                  stepResult = llvm::failure();
                  return;
                }
                MBarrierState &state = it->second;
                // We observed the barrier at this generation.
                generations[op] = state.generation;
                arriveMBarrier(threadIPs, generations, sleepingThreads, state);
                steppedThread = true;
              })
          .Case<mlir::barrier::MbarrierWaitOp>(
              [&](mlir::barrier::MbarrierWaitOp op) {
                // Wait operations _may_ wait on the barrier, depending on the
                // generation and the phase. Waiting on a barrier that has not
                // been initialized is an error (MB-Wait-Err).
                int barrierId = op.getBarrierId();
                auto it = mbarrierStates.find(barrierId);
                if (it == mbarrierStates.end()) {
                  llvm::errs() << "weft: wait on uninitialized mbarrier "
                               << barrierId << "\n";
                  stepResult = llvm::failure();
                  return;
                }
                MBarrierState &state = it->second;

                // `op.getCond()` is an `i1`; reading it zero-extended yields
                // 0/1.
                // (`ConstantIntOp::value()` sign-extends, so a 1-bit `true`
                // would come back as -1 instead of 1.)
                auto condAttr = llvm::cast<mlir::IntegerAttr>(
                    llvm::cast<mlir::arith::ConstantOp>(
                        op.getCond().getDefiningOp())
                        .getValue());
                int phase = condAttr.getValue().getZExtValue();

                // If the generation matches the phase % 2, then we need to
                // register ourselves as a waiter on this barrier. Otherwise, we
                // can proceed and say that we observed the barrier at the
                // current generation - 1.
                if (state.generation % 2 == phase) {
                  // We need to register ourselves as a waiter on this barrier
                  // and go to sleep until an arrival wakes us.
                  state.waiters.push_back(i);
                  sleepingThreads.insert(i);
                } else {
                  // We can proceed and say that we observed the barrier at the
                  // current generation - 1.
                  generations[op] = state.generation - 1;
                  steppedThread = true;
                }
              })
          .Case<mlir::barrier::NamedBarrierArriveOp>(
              [&](mlir::barrier::NamedBarrierArriveOp op) {
                NamedBarrierState &state =
                    namedBarrierStates[op.getBarrierId()];
                auto result = arriveNamedBarrier(threadIPs, generations,
                                                 sleepingThreads, state, op);
                if (failed(result)) {
                  stepResult = result;
                  return;
                }
                steppedThread = true;
              })
          .Case<mlir::barrier::NamedBarrierSyncOp>(
              [&](mlir::barrier::NamedBarrierSyncOp op) {
                NamedBarrierState &state =
                    namedBarrierStates[op.getBarrierId()];
                auto result = syncNamedBarrier(i, threadIPs, generations,
                                               sleepingThreads, state, op);
                if (failed(result)) {
                  stepResult = result;
                  return;
                }
                steppedThread = result.value();
              })
          .Case<mlir::barrier::SmemReadOp, mlir::barrier::SmemWriteOp>(
              [&](mlir::Operation *op) {
                // These operations are placeholders that don't do anything yet.
                steppedThread = true;
              })
          .Default([&](mlir::Operation *op) {
            llvm::report_fatal_error(
                "unhandled operation in thread program simulation: " +
                op->getName().getStringRef());
          });

      if (failed(stepResult)) {
        return stepResult;
      }

      // If we stepped a thread, then move on.
      if (steppedThread) {
        threadIPs[i] = op->getNextNode();
        break;
      }
    }
    // If we didn't step any threads and we're not done, then we've hit a
    // deadlock and the simulation of the thread programs has failed.
    if (!steppedThread && !areThreadsComplete()) {
      llvm::errs() << "weft: simulation of thread programs has failed due "
                      "to a deadlock\n";
      return llvm::failure();
    }
  }
  assert(areThreadsComplete());
  return llvm::success();
}

// Note (rohany): This implementation was written by Claude and looks
//  like it's going to be slow at larger program sizes.
// Computes the transitive closure of a relation over operations in place:
// repeatedly adds the edge (a, c) whenever (a, b) and (b, c) are both present,
// until no new edges are discovered.
static void transitiveClosure(
    std::set<std::pair<mlir::Operation *, mlir::Operation *>> &relation) {
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<std::pair<mlir::Operation *, mlir::Operation *>> toAdd;
    for (const auto &[a, b] : relation)
      for (const auto &[c, d] : relation)
        if (b == c && !relation.count({a, d}))
          toAdd.push_back({a, d});
    for (const auto &edge : toAdd)
      changed |= relation.insert(edge).second;
  }
}

// Returns the index of the thread program in `threadPrograms` that `op` belongs
// to. Raises a fatal error if `op` is not contained in any of them.
static int getThreadForOp(std::vector<mlir::func::FuncOp> &threadPrograms,
                          mlir::Operation *op) {
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  for (int i = 0, e = threadPrograms.size(); i < e; ++i)
    if (threadPrograms[i] == func)
      return i;
  llvm::report_fatal_error("operation does not belong to any thread program");
}

llvm::FailureOr<bool>
checkWellSynchronized(std::vector<mlir::func::FuncOp> &threadPrograms,
                      std::map<mlir::Operation *, int> &generations,
                      std::set<std::pair<mlir::Operation *, mlir::Operation *>>
                          &happensBeforeRelation) {
  int numThreads = threadPrograms.size();
  // Add a happens-before relation between each instruction in each thread
  // program.
  for (int i = 0; i < numThreads; i++) {
    mlir::Region &body = threadPrograms[i].getBody();
    mlir::Block &block = body.front();
    mlir::Operation *cur = &(*block.begin());
    while (cur != nullptr) {
      mlir::Operation *next = cur->getNextNode();
      if (next != nullptr) {
        happensBeforeRelation.insert({cur, next});
      }
      cur = next;
    }
  }

  // The following functions are helper functions for interacting with
  // mbarriers in the thread programs.

  // These functions implement the "Reg" and "Rel" sets in the algorithm for
  // mbarriers.
  auto getMBarrierArrivers = [&](int barrierId, int generation) {
    std::vector<mlir::Operation *> arrivers;
    for (int i = 0; i < numThreads; ++i) {
      for (auto arrive : threadPrograms[i]
                             .getBody()
                             .getOps<mlir::barrier::MbarrierArriveOp>()) {
        if (barrierId != static_cast<int>(arrive.getBarrierId()))
          continue;
        assert(generations.find(arrive) != generations.end());
        if (generations.at(arrive) == generation) {
          arrivers.push_back(arrive);
        }
      }
    }
    return arrivers;
  };
  // Collect every initialization of `barrierId` across the thread programs.
  auto getMBarrierInits = [&](int barrierId) {
    std::vector<mlir::Operation *> inits;
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::barrier::MbarrierInitOp init) {
        if (static_cast<int>(init.getBarrierId()) == barrierId)
          inits.push_back(init);
      });
    }
    return inits;
  };
  // The arrival count of a barrier comes from its (unique) initialization.
  auto getMBarrierArriverCount = [&](int barrierId) {
    auto inits = getMBarrierInits(barrierId);
    if (inits.empty()) {
      llvm::report_fatal_error("no barrier initialization found");
    }
    return size_t(
        llvm::cast<mlir::barrier::MbarrierInitOp>(inits.front())
            .getArrivalCount());
  };
  auto getMBarrierWaiters = [&](int barrierId, int generation) {
    std::vector<mlir::Operation *> waiters;
    for (int i = 0; i < numThreads; ++i) {
      for (auto wait : threadPrograms[i]
                           .getBody()
                           .getOps<mlir::barrier::MbarrierWaitOp>()) {
        if (barrierId != static_cast<int>(wait.getBarrierId()))
          continue;
        assert(generations.find(wait) != generations.end());
        if (generations.at(wait) == generation) {
          waiters.push_back(wait);
        }
      }
    }
    return waiters;
  };
  // Collect every use (arrive or wait, not init) of `barrierId` across the
  // thread programs.
  auto getMBarrierUses = [&](int barrierId) {
    std::vector<mlir::Operation *> uses;
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::Operation *op) {
        if (auto arrive = llvm::dyn_cast<mlir::barrier::MbarrierArriveOp>(op)) {
          if (static_cast<int>(arrive.getBarrierId()) == barrierId)
            uses.push_back(op);
        } else if (auto wait =
                       llvm::dyn_cast<mlir::barrier::MbarrierWaitOp>(op)) {
          if (static_cast<int>(wait.getBarrierId()) == barrierId)
            uses.push_back(op);
        }
      });
    }
    return uses;
  };

  // Collect all barrier ID's that were interacted with throughout
  // execution of the thread programs. Every used barrier must have been
  // initialized (an uninitialized use fails simulation), so the
  // initializations enumerate all barriers.
  auto getAllMBarrierIDs = [&]() {
    llvm::SetVector<int> barrierIds;
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::barrier::MbarrierInitOp op) {
        barrierIds.insert(static_cast<int>(op.getBarrierId()));
      });
    }
    return barrierIds;
  };

  // Collect all generations that a barrier was interacted with on
  // throughout the execution of the thread programs.
  auto getAllMBarrierGenerations = [&](int barrierId) {
    llvm::SetVector<int> barrierGenerations;
    // Record the simulated generation of `op` if it touches `barrierId`.
    auto collect = [&](mlir::Operation *op, int opBarrierId) {
      if (opBarrierId != barrierId)
        return;
      assert(generations.find(op) != generations.end());
      barrierGenerations.insert(generations.at(op));
    };
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::barrier::MbarrierArriveOp arrive) {
        collect(arrive, static_cast<int>(arrive.getBarrierId()));
      });
      thread.walk([&](mlir::barrier::MbarrierWaitOp wait) {
        collect(wait, static_cast<int>(wait.getBarrierId()));
      });
    }
    return barrierGenerations;
  };

  // The next set of functions are helper functions for interacting with
  // named barriers in the thread programs. They mirror the mbarrier
  // helpers above, with two differences specific to named barriers: a
  // named barrier's id is carried directly as an op attribute (rather
  // than on a defining op), and both `named_barrier_arrive` and
  // `named_barrier_sync` register an arrival while only
  // `named_barrier_sync` blocks (waits).

  // Extracts the named barrier id carried directly on an arrive/sync op.
  auto namedBarrierIdOf = [](mlir::Operation *op) -> int {
    if (auto arrive = llvm::dyn_cast<mlir::barrier::NamedBarrierArriveOp>(op))
      return static_cast<int>(arrive.getBarrierId());
    if (auto sync = llvm::dyn_cast<mlir::barrier::NamedBarrierSyncOp>(op))
      return static_cast<int>(sync.getBarrierId());
    llvm::report_fatal_error("operation is not a named barrier operation");
  };

  // These functions get the arrivers and syncers for a given named
  // barrier and generation.
  auto getNamedBarrierArrivers = [&](int barrierId, int generation) {
    std::vector<mlir::Operation *> arrivers;
    for (int i = 0; i < numThreads; ++i) {
      for (auto &opRef : threadPrograms[i].getBody().front()) {
        mlir::Operation *op = &opRef;
        if (!llvm::isa<mlir::barrier::NamedBarrierArriveOp,
                       mlir::barrier::NamedBarrierSyncOp>(op))
          continue;
        if (namedBarrierIdOf(op) != barrierId)
          continue;
        assert(generations.find(op) != generations.end());
        if (generations.at(op) == generation)
          arrivers.push_back(op);
      }
    }
    return arrivers;
  };

  auto getNamedBarrierWaiters = [&](int barrierId, int generation) {
    std::vector<mlir::Operation *> waiters;
    for (int i = 0; i < numThreads; ++i) {
      for (auto sync : threadPrograms[i]
                           .getBody()
                           .getOps<mlir::barrier::NamedBarrierSyncOp>()) {
        if (static_cast<int>(sync.getBarrierId()) != barrierId)
          continue;
        assert(generations.find(sync) != generations.end());
        if (generations.at(sync) == generation)
          waiters.push_back(sync);
      }
    }
    return waiters;
  };

  // Collect all named barrier ID's that were interacted with throughout
  // execution of the thread programs.
  auto getAllNamedBarrierIDs = [&]() {
    llvm::SetVector<int> barrierIds;
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::barrier::NamedBarrierArriveOp op) {
        barrierIds.insert(static_cast<int>(op.getBarrierId()));
      });
      thread.walk([&](mlir::barrier::NamedBarrierSyncOp op) {
        barrierIds.insert(static_cast<int>(op.getBarrierId()));
      });
    }
    return barrierIds;
  };

  // Collect all generations that a named barrier was interacted with on
  // throughout the execution of the thread programs.
  auto getAllNamedBarrierGenerations = [&](int barrierId) {
    llvm::SetVector<int> barrierGenerations;
    // Record the simulated generation of `op` if it touches `barrierId`.
    auto collect = [&](mlir::Operation *op, int opBarrierId) {
      if (opBarrierId != barrierId)
        return;
      assert(generations.find(op) != generations.end());
      barrierGenerations.insert(generations.at(op));
    };
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::barrier::NamedBarrierArriveOp arrive) {
        collect(arrive, static_cast<int>(arrive.getBarrierId()));
      });
      thread.walk([&](mlir::barrier::NamedBarrierSyncOp sync) {
        collect(sync, static_cast<int>(sync.getBarrierId()));
      });
    }
    return barrierGenerations;
  };

  // Add happens-before relationships between arrives and waits on the
  // same barrier at the same generation. This handles the direct cases
  // for mbarriers.
  for (auto barrierId : getAllMBarrierIDs()) {
    for (auto generation : getAllMBarrierGenerations(barrierId)) {
      for (auto arriver : getMBarrierArrivers(barrierId, generation)) {
        for (auto waiter : getMBarrierWaiters(barrierId, generation)) {
          happensBeforeRelation.insert({arriver, waiter});
        }
      }
    }
  }

  // TODO (rohany): This might be wrong ...
  // TODO (rohany): I think that we have to add the symmetric
  // happens-before
  //  relationships for mbarrier waits (just like named barrier syncs) as
  //  they also do resolve at the same time, just like named barrier
  //  syncs. I'll let this bake for a bit while getting confirmation from
  //  Ben.
  // for (auto barrierId : getAllMBarrierIDs()) {
  //   for (auto generation : getAllMBarrierGenerations(barrierId)) {
  //     for (auto c1 : getMBarrierWaiters(barrierId, generation)) {
  //       for (auto c2 : getMBarrierWaiters(barrierId, generation)) {
  //         if (c1 != c2) {
  //           happensBeforeRelation.insert({c1, c2});
  //           happensBeforeRelation.insert({c2, c1});
  //         }
  //       }
  //     }
  //   }
  // }

  // Next, add the appropriate happens-before relationships for named
  // barriers. The first direct happens-before relationship is between
  // arrivers at a particular generation and syncers at the same
  // generation.
  for (auto barrierId : getAllNamedBarrierIDs()) {
    for (auto generation : getAllNamedBarrierGenerations(barrierId)) {
      for (auto arriver : getNamedBarrierArrivers(barrierId, generation)) {
        for (auto syncer : getNamedBarrierWaiters(barrierId, generation)) {
          happensBeforeRelation.insert({arriver, syncer});
        }
      }
    }
  }

  // Next, there's a symmetric happens-before relationship between all
  // syncers at the same generation.
  for (auto barrierId : getAllNamedBarrierIDs()) {
    for (auto generation : getAllNamedBarrierGenerations(barrierId)) {
      for (auto c1 : getNamedBarrierWaiters(barrierId, generation)) {
        for (auto c2 : getNamedBarrierWaiters(barrierId, generation)) {
          if (c1 != c2) {
            happensBeforeRelation.insert({c1, c2});
            happensBeforeRelation.insert({c2, c1});
          }
        }
      }
    }
  }

  // Compute the transitive closure of the happens-before relation.
  transitiveClosure(happensBeforeRelation);

  // Check the initialization conditions for mbarriers (the `okUniqueInit`
  // and `okInit` checks of the mechanized algorithm).

  // Each mbarrier must be initialized by at most one `mbarrier_init` in the
  // whole CTA; repeated initialization/destruction cycles of an mbarrier are
  // deliberately out of scope. Note that this check is not directly
  // observable in weft today: with straight-line thread programs every
  // instruction executes, so a duplicate initialization always errs during
  // simulation (MB-Init-Err) before this check runs. It is implemented
  // anyway for clarity and to match the mechanization.
  for (auto barrierId : getAllMBarrierIDs()) {
    if (getMBarrierInits(barrierId).size() > 1) {
      llvm::errs() << "weft: mbarrier " << barrierId
                   << " has multiple initializations\n";
      return false;
    }
  }

  // Every use (arrive or wait) of an mbarrier must be ordered after the
  // barrier's initialization. A thread errs the moment its control reaches a
  // use with the barrier uninitialized, so the initialization must be forced
  // before the instruction that gates the use: its in-thread predecessor.
  // Anchoring at the use itself would be unsound, as a happens-before path
  // into the use may route through an arrive -> wait release edge, which
  // does not stop the use's thread from reaching the use early and erring.
  // A use with no in-thread predecessor is reachable at the initial
  // configuration -- where every mbarrier is uninitialized -- and rejects.
  for (auto barrierId : getAllMBarrierIDs()) {
    auto inits = getMBarrierInits(barrierId);
    assert(inits.size() == 1);
    mlir::Operation *init = inits.front();
    for (auto use : getMBarrierUses(barrierId)) {
      auto previousInstruction = use->getPrevNode();
      if (!previousInstruction) {
        llvm::errs() << "weft: use of mbarrier " << barrierId << " (thread "
                     << getThreadForOp(threadPrograms, use)
                     << ") is the first instruction of its thread; nothing "
                        "orders it after the barrier's initialization\n";
        return false;
      }
      // The initialization may itself be the use's predecessor; the
      // happens-before set carries no reflexive edges.
      if (init != previousInstruction &&
          !happensBeforeRelation.count({init, previousInstruction})) {
        llvm::errs() << "weft: initialization of mbarrier " << barrierId
                     << " (thread " << getThreadForOp(threadPrograms, init)
                     << ") does not happen-before use (thread "
                     << getThreadForOp(threadPrograms, use) << ")\n";
        return false;
      }
    }
  }

  // Finally, check the two cases of well-synchronization. We check
  // how far arrivals and waits can "drift" away from the
  // generations that were observed during the simulation.

  // This check will ensure that if we have an operation that arrives
  // on a barrier b at generation g, then there exists a happens-before
  // relationship between any arrival on b at g+1, to ensure that the
  // arrival at g+1 couldn't have happened until this operation finished.
  for (auto barrierId : getAllMBarrierIDs()) {
    for (auto generation : getAllMBarrierGenerations(barrierId)) {
      for (auto arriver : getMBarrierArrivers(barrierId, generation)) {
        for (auto nextArriver :
             getMBarrierArrivers(barrierId, generation + 1)) {
          // TODO (rohany): I'm not sure if this needs the preceding
          //  c3 check or not. It might be fine if we're just looking at
          //  arrivers.
          if (!happensBeforeRelation.count({arriver, nextArriver})) {
            llvm::errs() << "weft: arriver-arriver drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  arriver (thread "
                         << getThreadForOp(threadPrograms, arriver)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(arriver)
                         << "): " << "and arriver (thread "
                         << getThreadForOp(threadPrograms, nextArriver)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(nextArriver) << ") \n";
            return false;
          }
        }
      }
    }
  }

  // Now, bound the range that each wait operation can "drift" by.
  for (auto barrierId : getAllMBarrierIDs()) {
    for (auto generation : getAllMBarrierGenerations(barrierId)) {
      for (auto waiter : getMBarrierWaiters(barrierId, generation)) {

        // For backwards drifting, ensure that there is a happens-before
        // relationship between all arrives on the previous generation.
        // However, there's only certain generations where we need to do
        // this:
        //  1) If this wait was registered on generation -1, then there is
        //     no previous generation that we can search for, as the
        //     barrier was trivially skipped with a wait(b, 1) with b.g =
        //     0.
        //  2) If this wait was registered on generation 0, then there
        //  also
        //     is no previous set of arrives that we need to have
        //     synchronized against, as the barrier starts in an
        //     untriggered state at 0.
        // For generations that are after this, then we need to search
        // backwards.
        if (generation >= 1) {

          // If we don't have a previous instruction, then this is an
          // error.
          auto previousInstruction = waiter->getPrevNode();
          if (!previousInstruction) {
            llvm::errs() << "weft: waiter at generation >= 1 has no previous "
                            "instruction!\n";
            return false;
          }

          // If we have a previous instruction, then we error if there is
          // not an edge from the arrives in the previous generation to
          // the instruction before this wait.

          for (auto arriver : getMBarrierArrivers(barrierId, generation - 1)) {
            if (!happensBeforeRelation.count({arriver, previousInstruction})) {
              llvm::errs() << "weft: waiter at generation >= 1 has no "
                              "happens-before relationship with any arrives "
                              "at generation g-1!\n";
              return false;
            }
          }
        }

        // To avoid forwards drifting, there must be a happens-before
        // relationship from this waiter to at least one arriver on the
        // next generation.
        {
          auto arrivers = getMBarrierArrivers(barrierId, generation + 1);
          bool found = false;
          for (auto arriver : arrivers) {
            if (happensBeforeRelation.count({waiter, arriver})) {
              found = true;
              break;
            }
          }
          if (!found && arrivers.size() == getMBarrierArriverCount(barrierId)) {
            llvm::errs() << "weft: waiter-arriver drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  waiter (thread "
                         << getThreadForOp(threadPrograms, waiter)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(waiter)
                         << "): and any arriver on generation "
                         << (generation + 1) << "\n";
            return false;
          }
        }
      }
    }
  }

  // The final check is the original check in weft for drift of named
  // barriers. It checks that if there is a sync (c1) at generation k,
  // then there is a happens-before edge from c1 to all barrier operations
  // at generation k+1. The weft implementation checks this by making sure
  // that for all operations c2 that occur on k+1, then the preceeding
  // operation c3;c2 happens after c1. I believe this check is in place
  // because syncs on the same generation have happens-before
  // relationships with each other.
  for (auto barrierId : getAllNamedBarrierIDs()) {
    for (auto generation : getAllNamedBarrierGenerations(barrierId)) {
      // TODO (rohany): Also check all arrivers, as per the new algorithm.
      for (auto syncer : getNamedBarrierWaiters(barrierId, generation)) {
        // Check the next arrivers.
        for (auto arriver :
             getNamedBarrierArrivers(barrierId, generation + 1)) {
          auto c2 = arriver->getPrevNode();
          if (!c2 || !happensBeforeRelation.count({syncer, c2})) {
            llvm::errs() << "weft: syncer-arriver drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  syncer (thread "
                         << getThreadForOp(threadPrograms, syncer)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(syncer) << "): ";
            return false;
          }
        }
        // Check the next syncers.
        for (auto nextSyncer :
             getNamedBarrierWaiters(barrierId, generation + 1)) {
          auto c2 = nextSyncer->getPrevNode();
          if (!c2 || !happensBeforeRelation.count({syncer, c2})) {
            llvm::errs() << "weft: syncer-syncer drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  syncer (thread "
                         << getThreadForOp(threadPrograms, syncer)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(syncer) << "): ";
          }
        }
      }

      // Repeat this case for arrivers.
      for (auto arriver : getNamedBarrierArrivers(barrierId, generation)) {
        // Check the next arrivers.
        for (auto nextArriver :
             getNamedBarrierArrivers(barrierId, generation + 1)) {
          auto c2 = nextArriver->getPrevNode();
          if (!c2 || !happensBeforeRelation.count({arriver, c2})) {
            llvm::errs() << "weft: arriver-arriver drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  arriver(thread "
                         << getThreadForOp(threadPrograms, arriver)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(nextArriver) << "): ";
            return false;
          }
        }
        // Check the next syncers.
        for (auto syncer : getNamedBarrierWaiters(barrierId, generation + 1)) {
          auto c2 = syncer->getPrevNode();
          if (!c2 || !happensBeforeRelation.count({arriver, c2})) {
            llvm::errs() << "weft: arriver-syncer drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  arriver (thread "
                         << getThreadForOp(threadPrograms, arriver)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(syncer) << "): ";
          }
        }
      }
    }
  }

  return true;
}

// Detects data races among shared-memory accesses. Consumes the
// happens-before relation computed by `checkWellSynchronized` (after its
// transitive closure). Two accesses to the same address, at least one of
// which is a write, race if neither ordering between them is present in
// the happens-before relation.
llvm::FailureOr<bool>
checkForRaces(std::vector<mlir::func::FuncOp> &threadPrograms,
              std::set<std::pair<mlir::Operation *, mlir::Operation *>>
                  &happensBeforeRelation) {
  // Extracts the shared-memory address of an access. The address operand must
  // be produced by a compile-time `arith.constant`; otherwise we error out.
  auto getAddress = [](mlir::Operation *op) -> llvm::FailureOr<uint64_t> {
    return llvm::TypeSwitch<mlir::Operation *, llvm::FailureOr<uint64_t>>(op)
        .Case<mlir::barrier::SmemWriteOp, mlir::barrier::SmemReadOp>(
            [](auto access) -> llvm::FailureOr<uint64_t> {
              auto constant =
                  llvm::dyn_cast_if_present<mlir::arith::ConstantOp>(
                      access.getAddress().getDefiningOp());
              if (!constant)
                return access.emitOpError(
                    "shared-memory address must be a compile-time "
                    "arith.constant");
              auto addressAttr =
                  llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
              if (!addressAttr)
                return access.emitOpError(
                    "shared-memory address must be an integer arith.constant");
              return addressAttr.getValue().getZExtValue();
            })
        .Default([](mlir::Operation *op) -> llvm::FailureOr<uint64_t> {
          return op->emitOpError("operation is not a shared-memory access");
        });
  };

  // Collect every shared-memory access across all thread programs.
  struct MemAccess {
    mlir::Operation *op;
    bool isWrite;
    uint64_t address;

    MemAccess(mlir::Operation *op, bool isWrite, uint64_t address)
        : op(op), isWrite(isWrite), address(address) {}
  };
  std::vector<MemAccess> accesses;
  for (auto &thread : threadPrograms) {
    auto walkResult = thread.walk([&](mlir::Operation *op) -> mlir::WalkResult {
      bool isWrite = llvm::isa<mlir::barrier::SmemWriteOp>(op);
      if (!isWrite && !llvm::isa<mlir::barrier::SmemReadOp>(op))
        return mlir::WalkResult::advance();
      auto address = getAddress(op);
      if (failed(address))
        return mlir::WalkResult::interrupt();
      accesses.emplace_back(op, isWrite, *address);
      return mlir::WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return mlir::failure();
  }

  // For every pair of accesses to the same address where at least one is
  // a write, ensure the pair is ordered in at least one direction by the
  // happens-before relation. If neither ordering is present, the accesses
  // are concurrent and therefore race. We report every race rather than
  // stopping at the first one.
  bool foundRace = false;
  for (size_t i = 0; i < accesses.size(); ++i) {
    for (size_t j = i + 1; j < accesses.size(); ++j) {
      const MemAccess &a = accesses[i];
      const MemAccess &b = accesses[j];
      if (!a.isWrite && !b.isWrite)
        continue;
      if (a.address != b.address)
        continue;
      if (happensBeforeRelation.count({a.op, b.op}) ||
          happensBeforeRelation.count({b.op, a.op}))
        continue;

      foundRace = true;
      llvm::errs() << "weft: race detected on address " << a.address
                   << " between:\n";
      llvm::errs() << "  " << (a.isWrite ? "write" : "read") << " (thread "
                   << getThreadForOp(threadPrograms, a.op) << "): ";
      a.op->print(llvm::errs());
      llvm::errs() << "\n  " << (b.isWrite ? "write" : "read") << " (thread "
                   << getThreadForOp(threadPrograms, b.op) << "): ";
      b.op->print(llvm::errs());
      llvm::errs() << "\n";
    }
  }

  return !foundRace;
}

mlir::LogicalResult
runFullWellSyncPipeline(std::vector<mlir::func::FuncOp> &threadPrograms) {
  std::map<mlir::Operation *, int> generations;
  if (mlir::failed(simulateThreadPrograms(threadPrograms, generations))) {
    llvm::errs() << "weft: failed to simulate the thread programs\n";
    return mlir::failure();
  }

  std::set<std::pair<mlir::Operation *, mlir::Operation *>>
      happensBeforeRelation;
  llvm::FailureOr<bool> wellSynchronizedResult =
      checkWellSynchronized(threadPrograms, generations, happensBeforeRelation);
  if (mlir::failed(wellSynchronizedResult)) {
    llvm::errs()
        << "weft: failed to check if the program is well-synchronized\n";
    return mlir::failure();
  }
  if (!wellSynchronizedResult.value()) {
    llvm::errs() << "weft: the program is not well-synchronized\n";
    return mlir::failure();
  }

  llvm::FailureOr<bool> racesResult =
      checkForRaces(threadPrograms, happensBeforeRelation);
  if (mlir::failed(racesResult)) {
    llvm::errs() << "weft: failed to check for races\n";
    return mlir::failure();
  }
  if (!racesResult.value()) {
    llvm::errs() << "weft: the program has a race\n";
    return mlir::failure();
  }

  llvm::outs() << "weft: the program is race-free\n";
  return mlir::success();
}

static llvm::cl::SubCommand
    StandardCmd("standard", "Run the standard weft race-detection pipeline");
static llvm::cl::SubCommand SingleNestedLoopCmd(
    "single-nested-loop",
    "Analyze programs with a single nested loop (not yet implemented)");

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input mlir file>"),
                  llvm::cl::Required, llvm::cl::sub(StandardCmd),
                  llvm::cl::sub(SingleNestedLoopCmd));

// Length of the program to unroll. Parsed via the LLVM command-line
// infrastructure; reserved for driving loop unrolling in the pipeline.
static llvm::cl::opt<int>
    unrollLength("n", llvm::cl::desc("Length of the program to unroll."),
                 llvm::cl::value_desc("n"), llvm::cl::init(0),
                 llvm::cl::sub(StandardCmd),
                 llvm::cl::sub(SingleNestedLoopCmd));

// When set, print the thread-specialized programs (the module after the pass
// pipeline) to stdout. Off by default so that only the analysis result is
// emitted.
static llvm::cl::opt<bool> printThreadPrograms(
    "print-thread-programs",
    llvm::cl::desc("Print the thread-specialized programs to stdout."),
    llvm::cl::init(false), llvm::cl::sub(StandardCmd),
    llvm::cl::sub(SingleNestedLoopCmd));

llvm::FailureOr<std::vector<mlir::func::FuncOp>>
collectThreadPrograms(mlir::ModuleOp module, int numThreads) {
  std::vector<mlir::func::FuncOp> threadPrograms(numThreads);
  module.walk([&](mlir::func::FuncOp func) {
    mlir::IntegerAttr threadSpecializedAttr = llvm::dyn_cast<mlir::IntegerAttr>(
        func->getAttr(mlir::barrier::kThreadSpecializedAttrName));
    if (!threadSpecializedAttr)
      return;
    int tid = threadSpecializedAttr.getInt();
    threadPrograms[tid] = func;
  });
  for (int i = 0; i < numThreads; i++) {
    if (!threadPrograms[i]) {
      return module.emitOpError() << "thread program " << i << " is not found";
    }
  }
  return threadPrograms;
}

static int runStandardWeft(mlir::ModuleOp module) {
  llvm::FailureOr<int> numThreadsResult = getNumThreads(module);
  if (mlir::failed(numThreadsResult)) {
    llvm::errs() << "weft: failed to get the number of threads from "
                    "input module\n";
    return 1;
  }
  int numThreads = numThreadsResult.value();

  // Specialize the functions's `index` argument to the requested `n`
  // before running the pipeline.
  if (mlir::failed(specializeEntryArguments(module, unrollLength))) {
    llvm::errs() << "weft: failed to specialize the entry argument\n";
    return 1;
  }

  // Build the pass pipeline: canonicalize, then
  // barrier-thread-specialize.
  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::barrier::createFullUnrollLoopsPass());
  pm.addPass(mlir::barrier::createBarrierThreadSpecializePass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "weft: pass pipeline failed\n";
    return 1;
  }

  if (printThreadPrograms)
    module.print(llvm::outs());

  // Collect all thread programs from the input specialized module.
  llvm::FailureOr<std::vector<mlir::func::FuncOp>> threadProgramsResult =
      collectThreadPrograms(module, numThreads);
  if (mlir::failed(threadProgramsResult)) {
    llvm::errs() << "weft: failed to collect thread programs\n";
    return 1;
  }
  std::vector<mlir::func::FuncOp> threadPrograms =
      std::move(threadProgramsResult.value());

  if (mlir::failed(runFullWellSyncPipeline(threadPrograms)))
    return 1;

  return 0;
}

mlir::LogicalResult
checkWellFormedSingleNestedLoopProgram(mlir::func::FuncOp program) {
  if (!program.getBody().hasOneBlock()) {
    return program.emitOpError()
           << "well-formed single-nested-loop program requires exactly one "
              "block in the body";
  }

  auto checkNormalInstruction =
      [&](mlir::Operation *op, llvm::StringRef region) -> mlir::LogicalResult {
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
        .Case<mlir::arith::ConstantOp>([&](auto) { return mlir::success(); })
        .Case<mlir::barrier::NamedBarrierArriveOp,
              mlir::barrier::NamedBarrierSyncOp, mlir::barrier::SmemReadOp,
              mlir::barrier::SmemWriteOp>([&](auto) { return mlir::success(); })
        .Case<mlir::barrier::MbarrierInitOp, mlir::barrier::MbarrierArriveOp,
              mlir::barrier::MbarrierWaitOp>([&](auto) {
          return op->emitOpError()
                 << "mbarrier operations are not yet supported in " << region;
        })
        .Case<mlir::scf::ForOp>([&](auto) {
          return op->emitOpError()
                 << "nested loops are not supported in " << region;
        })
        .Case<mlir::scf::IfOp>([&](auto) {
          return op->emitOpError()
                 << "conditional control flow is not supported in " << region;
        })
        .Default([&](mlir::Operation *) {
          return op->emitOpError()
                 << "unsupported operation in " << region
                 << "; expected arith.constant, named barrier, or "
                    "shared-memory operations";
        });
  };

  mlir::Block &block = program.getBody().front();
  enum class Phase { Prefix, Epilogue };
  Phase phase = Phase::Prefix;
  bool seenLoop = false;

  for (mlir::Operation &op : block) {
    if (mlir::isa<mlir::func::ReturnOp>(op))
      break;

    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      if (seenLoop) {
        return forOp.emitOpError()
               << "expected at most one scf.for in the program";
      }
      if (phase == Phase::Epilogue) {
        return forOp.emitOpError() << "scf.for must appear before the epilogue";
      }

      if (!forOp.getRegion().hasOneBlock()) {
        return forOp.emitOpError()
               << "scf.for body must contain exactly one block";
      }

      mlir::Block &loopBody = forOp.getRegion().front();
      for (mlir::Operation &bodyOp : loopBody) {
        if (&bodyOp == loopBody.getTerminator())
          break;
        if (mlir::failed(checkNormalInstruction(&bodyOp, "loop body")))
          return mlir::failure();
      }

      if (!mlir::isa<mlir::scf::YieldOp>(loopBody.getTerminator())) {
        return loopBody.getTerminator()->emitOpError()
               << "scf.for body must terminate with scf.yield";
      }

      seenLoop = true;
      phase = Phase::Epilogue;
      continue;
    }

    llvm::StringRef region = phase == Phase::Prefix ? "prefix" : "epilogue";
    if (mlir::failed(checkNormalInstruction(&op, region)))
      return mlir::failure();
  }

  if (!seenLoop) {
    return program.emitOpError()
           << "expected exactly one scf.for in the program";
  }

  return mlir::success();
}

static mlir::FailureOr<mlir::scf::ForOp>
findSingleForLoop(mlir::func::FuncOp program) {
  mlir::scf::ForOp loop;
  for (mlir::Operation &op : program.getBody().front()) {
    if (mlir::isa<mlir::func::ReturnOp>(op))
      break;
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      if (loop)
        return program.emitOpError()
               << "expected at most one scf.for in the program";
      loop = forOp;
    }
  }
  if (!loop)
    return program.emitOpError()
           << "expected exactly one scf.for in the program";
  return loop;
}

// Returns f(b) from Theorem ik-loop-unrolling for the given per-iteration
// arrival totals and configured arrival count of barrier b.
static int computeBarrierUnrollFactor(int arrivers, int arrivalCount) {
  if (arrivers == arrivalCount)
    return 1;
  if (arrivers != 0 && arrivalCount % arrivers == 0)
    return arrivalCount / arrivers;
  if (arrivalCount != 0 && arrivers % arrivalCount == 0)
    return 1;
  return std::lcm(arrivers, arrivalCount);
}

llvm::FailureOr<int>
computeKForThreadLoops(std::vector<mlir::func::FuncOp> &threadPrograms) {
  struct BarrierInfo {
    int arrivers = 0;
    std::optional<int> arrivalCount;
  };
  std::map<int, BarrierInfo> barrierInfos;

  auto recordNamedBarrierArrival =
      [&](mlir::Operation *op, int barrierId,
          int arrivalCount) -> mlir::LogicalResult {
    BarrierInfo &info = barrierInfos[barrierId];
    info.arrivers++;
    if (!info.arrivalCount.has_value())
      info.arrivalCount = arrivalCount;
    if (info.arrivalCount.value() != arrivalCount) {
      return op->emitOpError()
             << "inconsistent arrival count for named barrier " << barrierId
             << "; expected " << info.arrivalCount.value() << ", but found "
             << arrivalCount;
    }
    return mlir::success();
  };

  for (mlir::func::FuncOp program : threadPrograms) {
    mlir::FailureOr<mlir::scf::ForOp> loopResult = findSingleForLoop(program);
    if (mlir::failed(loopResult))
      return mlir::failure();

    mlir::Block &loopBody = loopResult->getRegion().front();
    for (mlir::Operation &op : loopBody) {
      if (&op == loopBody.getTerminator())
        break;

      if (auto arrive =
              mlir::dyn_cast<mlir::barrier::NamedBarrierArriveOp>(&op)) {
        if (mlir::failed(recordNamedBarrierArrival(
                &op, static_cast<int>(arrive.getBarrierId()),
                static_cast<int>(arrive.getArrivalCount()))))
          return mlir::failure();
        continue;
      }

      if (auto sync = mlir::dyn_cast<mlir::barrier::NamedBarrierSyncOp>(&op)) {
        if (mlir::failed(recordNamedBarrierArrival(
                &op, static_cast<int>(sync.getBarrierId()),
                static_cast<int>(sync.getArrivalCount()))))
          return mlir::failure();
      }
    }
  }

  int k = 1;
  for (const auto &it : barrierInfos) {
    const auto &info = it.second;
    assert(info.arrivalCount.has_value());
    k = std::lcm(k, computeBarrierUnrollFactor(info.arrivers,
                                               info.arrivalCount.value()));
  }

  return k;
}

static int runSingleNestedLoopWeft(mlir::ModuleOp module) {
  llvm::FailureOr<int> numThreadsResult = getNumThreads(module);
  if (mlir::failed(numThreadsResult)) {
    llvm::errs() << "weft: failed to get the number of threads from "
                    "input module\n";
    return 1;
  }
  int numThreads = numThreadsResult.value();

  // Just run the thread specializer, since we aren't going to fully
  // unroll our loops (yet).

  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::barrier::createBarrierThreadSpecializePass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "weft: pass pipeline failed\n";
    return 1;
  }

  if (printThreadPrograms)
    module.print(llvm::outs());

  // Collect the programs for each thread.
  llvm::FailureOr<std::vector<mlir::func::FuncOp>> threadProgramsResult =
      collectThreadPrograms(module, numThreads);
  if (mlir::failed(threadProgramsResult)) {
    llvm::errs() << "weft: failed to collect thread programs\n";
    return 1;
  }
  std::vector<mlir::func::FuncOp> threadPrograms =
      std::move(threadProgramsResult.value());

  // 3 steps for the verification here:

  // 1) Check that the program is indeed of the form P; L; E.
  for (mlir::func::FuncOp threadProgram : threadPrograms) {
    if (mlir::failed(checkWellFormedSingleNestedLoopProgram(threadProgram))) {
      llvm::errs() << "weft: thread program is not well-formed\n";
      return 1;
    }
  }

  // 2) Compute K for the loop.
  // TODO (rohany): The computation of K needs to also make sure that
  //  k is LCM'd against the number that ensures SMEM accesses are also
  //  periodic. However, if the main loop doesn't have any control flow
  //  left in it anymore, then this value should just be 1.
  llvm::FailureOr<int> kResult = computeKForThreadLoops(threadPrograms);
  if (mlir::failed(kResult)) {
    llvm::errs() << "weft: failed to compute K for thread loops\n";
    return 1;
  }

  int k = kResult.value();
  llvm::outs() << "weft: computed K for program: " << k << "\n";

  // TODO (rohany): I can make this exactly 2k now, thanks to the proof in
  // lean.
  //  However, the algorithm needs to also check that the prefix of the loop
  //  is well-synchronized by itself, which is a little unclean.
  // 3) Unroll the program and check well-sync for each k in the range [0,
  // 3K].
  //  We'll hijack the logic for argument specialization here.
  for (int i = 0; i < 3 * k; i++) {
    llvm::outs() << "weft: checking well-synchronized and race-free at k = "
                 << i << "\n";
    // Clone the module.
    auto newMod = module.clone();
    // Specialize the functions's `index` argument to the requested `n`.
    if (mlir::failed(specializeEntryArguments(newMod, i))) {
      llvm::errs() << "weft: failed to specialize for " << i << "\n";
      return 1;
    }

    // Unroll the program.
    mlir::PassManager pm(newMod.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::barrier::createFullUnrollLoopsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(newMod))) {
      llvm::errs() << "weft: pass pipeline failed\n";
      return 1;
    }

    // Extract the thread programs from the cloned module.
    llvm::FailureOr<std::vector<mlir::func::FuncOp>> newThreadPrograms =
        collectThreadPrograms(newMod, numThreads);
    if (mlir::failed(newThreadPrograms)) {
      llvm::errs() << "weft: failed to collect thread programs\n";
      return 1;
    }

    // Now, check well synchronized!
    if (mlir::failed(runFullWellSyncPipeline(newThreadPrograms.value()))) {
      llvm::errs() << "weft: failed to check well synchronized for " << i
                   << "\n";
      return 1;
    }
  }

  llvm::outs() << "weft: input program is well-synchronized and race-free "
                  "for all n!\n";

  return 0;
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "weft barrier race detection\n"
      "  Usage: weft <command> <input.mlir> [options]\n"
      "  Commands: standard, single-nested-loop\n");

  if (!StandardCmd && !SingleNestedLoopCmd) {
    llvm::errs() << "weft: please specify a command (standard or "
                    "single-nested-loop)\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::barrier::BarrierDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "weft: failed to parse '" << inputFilename << "'\n";
    return 1;
  }

  if (StandardCmd)
    return runStandardWeft(*module);
  if (SingleNestedLoopCmd)
    return runSingleNestedLoopWeft(*module);

  llvm_unreachable("unhandled weft command");
}
