//===- weft.cpp - Barrier thread-specialization driver --------*- C++ -*-===//
//
// Reads an MLIR file, runs a fixed pass pipeline (canonicalize followed by
// barrier-thread-specialize) over it, and prints the result. Unlike
// barrier-opt (which is a general mlir-opt clone), weft hard-codes the
// pipeline and exposes a single tuning knob `n`.
//
//===----------------------------------------------------------------------===//

#include "barrier/Dialect/BarrierDialect.h"
#include "barrier/Dialect/BarrierOps.h"
#include "barrier/Transform/BarrierPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

#include <set>

// Specializes the single function in `module` by replacing its lone `index`
// argument with the compile-time constant `n`, then dropping the now-unused
// argument from the function signature. The module is expected to contain
// exactly one function taking a single `index` argument.
static mlir::LogicalResult specializeEntryArgument(mlir::ModuleOp module,
                                                   int n) {
  auto funcs = llvm::to_vector(module.getOps<mlir::func::FuncOp>());
  if (funcs.size() != 1)
    return module.emitOpError()
           << "requires exactly one function in the module, but found "
           << funcs.size();
  mlir::func::FuncOp func = funcs.front();

  // A function with no arguments has already had its unroll constants inlined,
  // so there is nothing left to specialize.
  if (func.getNumArguments() == 0)
    return mlir::success();

  if (func.getNumArguments() != 1)
    return func.emitOpError() << "requires zero or one argument, but found "
                              << func.getNumArguments();

  mlir::BlockArgument arg = func.getArgument(0);
  if (!mlir::isa<mlir::IndexType>(arg.getType()))
    return func.emitOpError()
           << "requires its argument to be of type 'index', but found "
           << arg.getType();

  if (n <= 0)
    return func.emitOpError()
           << "requires the unroll length 'n' to be positive, but found " << n;

  // Materialize the constant at the start of the entry block and rewrite every
  // use of the argument to refer to it instead.
  mlir::Block &entry = func.getBody().front();
  mlir::OpBuilder b(&entry, entry.begin());
  auto constant = mlir::arith::ConstantOp::create(
      b, func.getLoc(), b.getIndexType(), b.getIndexAttr(n));
  arg.replaceAllUsesWith(constant.getResult());

  // The argument is now dead; remove it from the block and function type.
  bool erased = succeeded(func.eraseArgument(0));
  assert(erased && "failed to erase the function argument");
  (void)erased;

  return mlir::success();
}

llvm::FailureOr<int> getNumThreads(mlir::ModuleOp module) {
  auto funcs = llvm::to_vector(module.getOps<mlir::func::FuncOp>());
  if (funcs.size() != 1)
    return module.emitOpError()
           << "requires exactly one function in the module, but found "
           << funcs.size();
  mlir::func::FuncOp func = funcs.front();
  mlir::IntegerAttr numThreadsAttr = llvm::dyn_cast<mlir::IntegerAttr>(
      func->getAttr(mlir::barrier::kNumThreadsAttrName));
  if (!numThreadsAttr)
    return func.emitOpError() << "input function therequires \""
                              << mlir::barrier::kNumThreadsAttrName
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

  // TODO (rohany): This simulation approach works right now assuming that all
  //  barriers have an arrival count of 1. When this changes, the simulation
  //  infrastructure will need to be updated.
  struct BarrierState {
    int generation = 0;
    int arrivalCount = 1;
    std::vector<int> waiters;
  };

  // Register an arrival on `barrier`. When the last expected arrival lands, the
  // barrier flips: every waiting thread observes the current generation, is
  // advanced past its wait, and the generation is bumped for the next phase.
  auto arriveBarrier = [](std::vector<mlir::Operation *> &threadIPs,
                          std::map<mlir::Operation *, int> &generations,
                          std::set<int> &sleepingThreads,
                          BarrierState &barrier) {
    barrier.arrivalCount--;
    if (barrier.arrivalCount == 0) {
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
      barrier.arrivalCount = 1;
    }
  };

  // Barrier ID's to current generations.
  std::map<int, BarrierState> barrierStates;
  // Collect all barrier ID's from all of the thread programs.
  for (auto &func : threadPrograms) {
    func.walk([&](mlir::barrier::NewOp op) {
      // Barriers are actually initialized to generation 0.
      if (barrierStates.find(op.getBarrierId()) != barrierStates.end()) {
        return;
      }
      barrierStates.insert({op.getBarrierId(), BarrierState{}});
    });
  }

  auto areThreadsComplete = [&]() {
    for (int i = 0; i < numThreads; i++) {
      if (threadIPs[i] != threadPrograms[i].getBody().front().getTerminator()) {
        return false;
      }
    }
    return true;
  };

  // Diagnostic helper: dump the state of every barrier (generation, outstanding
  // arrival count, and the set of threads waiting on it). `when` labels the
  // dump relative to the step being executed (e.g. "before" / "after").
  auto dumpBarrierStates = [&](llvm::StringRef when) {
    llvm::outs() << "    barriers " << when << ":";
    if (barrierStates.empty())
      llvm::outs() << " <none>";
    for (auto &[barrierId, state] : barrierStates) {
      llvm::outs() << " [id=" << barrierId << " gen=" << state.generation
                   << " arrivalCount=" << state.arrivalCount << " waiters={";
      for (size_t w = 0; w < state.waiters.size(); ++w)
        llvm::outs() << (w == 0 ? "" : ",") << state.waiters[w];
      llvm::outs() << "}]";
    }
    llvm::outs() << "\n";
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
    // Try to step a thread. We'll proceed in round-robin order.
    for (int i = 0; i < numThreads; i++) {
      // Skip threads that are asleep on a barrier; only an arrival on that
      // barrier can wake them.
      if (sleepingThreads.count(i))
        continue;

      mlir::Operation *op = threadIPs[i];

      // llvm::outs() << "weft: attempting to step thread " << i
      //              << ", executing: ";
      // op->print(llvm::outs());
      // llvm::outs() << "\n";
      // dumpBarrierStates("before");

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
          .Case<mlir::barrier::NewOp>([&](mlir::barrier::NewOp op) {
            // New barrier operations also don't "do" anything.
            steppedThread = true;
          })
          .Case<mlir::barrier::ArriveOp>([&](mlir::barrier::ArriveOp op) {
            // Arrive operations will read the current generation and advance
            // the generation.
            int barrierId = llvm::dyn_cast<mlir::barrier::NewOp>(
                                op.getBarrier().getDefiningOp())
                                .getBarrierId();
            BarrierState &state = barrierStates.at(barrierId);
            // We observed the barrier at this generation.
            generations[op] = state.generation;
            arriveBarrier(threadIPs, generations, sleepingThreads, state);
            steppedThread = true;
          })
          .Case<mlir::barrier::WaitOp>([&](mlir::barrier::WaitOp op) {
            // Wait operations _may_ wait on the barrier, depending on the
            // generation and the phase.
            int barrierId = llvm::dyn_cast<mlir::barrier::NewOp>(
                                op.getBarrier().getDefiningOp())
                                .getBarrierId();
            BarrierState &state = barrierStates.at(barrierId);

            // `op.getCond()` is an `i1`; reading it zero-extended yields 0/1.
            // (`ConstantIntOp::value()` sign-extends, so a 1-bit `true` would
            // come back as -1 instead of 1.)
            auto condAttr = llvm::cast<mlir::IntegerAttr>(
                llvm::cast<mlir::arith::ConstantOp>(
                    op.getCond().getDefiningOp())
                    .getValue());
            int phase = condAttr.getValue().getZExtValue();

            // If the generation matches the phase % 2, then we need to register
            // ourselves as a waiter on this barrier. Otherwise, we can proceed
            // and say that we observed the barrier at the current generation
            // - 1.
            if (state.generation % 2 == phase) {
              // We need to register ourselves as a waiter on this barrier and
              // go to sleep until an arrival wakes us.
              state.waiters.push_back(i);
              sleepingThreads.insert(i);
            } else {
              // We can proceed and say that we observed the barrier at the
              // current generation - 1.
              generations[op] = state.generation - 1;
              steppedThread = true;
            }
          })
          .Case<mlir::barrier::TMALoadOp, mlir::barrier::TCMMAOp>(
              [&](mlir::Operation *op) {
                // These operations are placeholders that don't do anything yet.
                steppedThread = true;
              })
          .Default([&](mlir::Operation *op) {
            llvm::report_fatal_error(
                "unhandled operation in thread program simulation: " +
                op->getName().getStringRef());
          });

      // dumpBarrierStates("after");

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
                      std::map<mlir::Operation *, int> &generations) {
  int numThreads = threadPrograms.size();
  std::set<std::pair<mlir::Operation *, mlir::Operation *>>
      happensBeforeRelation;
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

  // These functions implement the "Reg" and "Rel" sets in the algorithm.
  auto getArrivers = [&](int barrierId, int generation) {
    std::vector<mlir::Operation *> arrivers;
    for (int i = 0; i < numThreads; ++i) {
      for (auto arrive :
           threadPrograms[i].getBody().getOps<mlir::barrier::ArriveOp>()) {
        auto arriveBarrierId = arrive.getBarrier()
                                   .getDefiningOp<mlir::barrier::NewOp>()
                                   .getBarrierId();
        if (barrierId != arriveBarrierId)
          continue;
        assert(generations.find(arrive) != generations.end());
        if (generations.at(arrive) == generation) {
          arrivers.push_back(arrive);
        }
      }
    }
    return arrivers;
  };
  auto getWaiters = [&](int barrierId, int generation) {
    std::vector<mlir::Operation *> waiters;
    for (int i = 0; i < numThreads; ++i) {
      for (auto wait :
           threadPrograms[i].getBody().getOps<mlir::barrier::WaitOp>()) {
        auto waitBarrierId = wait.getBarrier()
                                 .getDefiningOp<mlir::barrier::NewOp>()
                                 .getBarrierId();
        if (barrierId != waitBarrierId)
          continue;
        assert(generations.find(wait) != generations.end());
        if (waitBarrierId == barrierId && generations.at(wait) == generation) {
          waiters.push_back(wait);
        }
      }
    }
    return waiters;
  };

  // Collect all barrier ID's that were interacted with throughout execution
  // of the thread programs.
  auto getAllBarrierIds = [&]() {
    llvm::SetVector<int> barrierIds;
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::barrier::NewOp op) {
        barrierIds.insert(op.getBarrierId());
      });
    }
    return barrierIds;
  };

  // Collect all generations that a barrier was interacted with on throughout
  // the execution of the thread programs.
  auto getAllGenerations = [&](int barrierId) {
    llvm::SetVector<int> barrierGenerations;
    // Record the simulated generation of `op` if it touches `barrierId`.
    auto collect = [&](mlir::Operation *op, int opBarrierId) {
      if (opBarrierId != barrierId)
        return;
      assert(generations.find(op) != generations.end());
      barrierGenerations.insert(generations.at(op));
    };
    for (auto &thread : threadPrograms) {
      thread.walk([&](mlir::barrier::ArriveOp arrive) {
        collect(arrive, arrive.getBarrier()
                            .getDefiningOp<mlir::barrier::NewOp>()
                            .getBarrierId());
      });
      thread.walk([&](mlir::barrier::WaitOp wait) {
        collect(wait, wait.getBarrier()
                          .getDefiningOp<mlir::barrier::NewOp>()
                          .getBarrierId());
      });
    }
    return barrierGenerations;
  };

  // Add happens-before relationships between arrives and waits on the same
  // barrier at the same generation.
  for (auto barrierId : getAllBarrierIds()) {
    for (auto generation : getAllGenerations(barrierId)) {
      for (auto arriver : getArrivers(barrierId, generation)) {
        for (auto waiter : getWaiters(barrierId, generation)) {
          happensBeforeRelation.insert({arriver, waiter});
        }
      }
    }
  }

  // Compute the transitive closure of the happens-before relation.
  transitiveClosure(happensBeforeRelation);

  // Finally, check the two cases of well-synchronization. We check
  // how far arrivals and waits can "drift" away from the
  // generations that were observed during the simulation.

  // This check will ensure that if we have an operation that arrives
  // on a barrier b at generation g, then there exists a happens-before
  // relationship between any arrival on b at g+1, to ensure that the
  // arrival at g+1 couldn't have happened until this operation finished.
  for (auto barrierId : getAllBarrierIds()) {
    for (auto generation : getAllGenerations(barrierId)) {
      for (auto arriver : getArrivers(barrierId, generation)) {
        for (auto nextArriver : getArrivers(barrierId, generation + 1)) {
          if (!happensBeforeRelation.count({arriver, nextArriver})) {
            llvm::errs() << "weft: arriver-arriver drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  arriver (thread "
                         << getThreadForOp(threadPrograms, arriver)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(arriver) << "): ";
            return false;
          }
        }
      }
    }
  }

  // Now, bound the range that each wait operation can "drift" by.
  for (auto barrierId : getAllBarrierIds()) {
    for (auto generation : getAllGenerations(barrierId)) {
      for (auto waiter : getWaiters(barrierId, generation)) {

        // For backwards drifting, ensure that there is a happens-before
        // relationship between all arrives on the previous generation. However,
        // there's only certain generations where we need to do this:
        //  1) If this wait was registered on generation -1, then there is
        //     no previous generation that we can search for, as the barrier
        //     was trivially skipped with a wait(b, 1) with b.g = 0.
        //  2) If this wait was registered on generation 0, then there also
        //     is no previous set of arrives that we need to have synchronized
        //     against, as the barrier starts in an untriggered state at 0.
        // For generations that are after this, then we need to search
        // backwards.
        if (generation >= 1) {
          // There are few checks to make here. First, if there are no
          // operations at all that happen before this wait at a generation
          // higher than 0, then this wait can clearly drift earlier.
          // TODO (rohany): A more efficient happens-before relation data
          //  structure would make this more efficient.
          {
            bool found = false;
            for (auto &it : happensBeforeRelation) {
              if (it.second == waiter) {
                found = true;
                break;
              }
            }
            if (!found) {
              llvm::errs()
                  << "weft: waiter at generation >= 1 has no predecessors!\n";
              return false;
            }
          }
          // Next, there should be a happens-before relationship between all
          // arrives at generation g-1 and this wait, otherwise this wait could
          // have snuck earlier and waited on an earlier generation.
          for (auto arriver : getArrivers(barrierId, generation - 1)) {
            if (!happensBeforeRelation.count({arriver, waiter})) {
              llvm::errs() << "weft: waiter at generation >= 1 has no "
                              "happens-before relationship with any arrives at "
                              "generation g-1!\n";
              return false;
            }
          }
        }

        // For forwards drifting, there must be a happens-before relationship
        // between waiter and all arrivers on the next generation.
        for (auto arriver : getArrivers(barrierId, generation + 1)) {
          if (!happensBeforeRelation.count({waiter, arriver})) {
            llvm::errs() << "weft: waiter-arriver drift! there is no "
                            "happens-before relationship between:\n";
            llvm::errs() << "\n  waiter (thread "
                         << getThreadForOp(threadPrograms, waiter)
                         << ", barrier " << barrierId << ", generation "
                         << generations.at(waiter) << "): ";
            return false;
          }
        }
      }
    }
  }

  return true;
}

int main(int argc, char **argv) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input mlir file>"));

  // Length of the program to unroll. Parsed via the LLVM command-line
  // infrastructure; reserved for driving loop unrolling in the pipeline.
  static llvm::cl::opt<int> unrollLength(
      "n", llvm::cl::desc("Length of the program to unroll."),
      llvm::cl::value_desc("n"), llvm::cl::init(0));

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "weft race detection\n");
  if (inputFilename.empty()) {
    llvm::errs() << "weft: please provide an input filename!\n";
    return 1;
  }

  // Register the dialects weft needs to parse and transform the input.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::barrier::BarrierDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse the input MLIR module.
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "weft: failed to parse '" << inputFilename << "'\n";
    return 1;
  }

  llvm::FailureOr<int> numThreadsResult = getNumThreads(*module);
  if (mlir::failed(numThreadsResult)) {
    llvm::errs()
        << "weft: failed to get the number of threads from input module\n";
    return 1;
  }
  int numThreads = numThreadsResult.value();

  // Specialize the function's `index` argument to the requested `n` before
  // running the pipeline.
  if (mlir::failed(specializeEntryArgument(*module, unrollLength))) {
    llvm::errs() << "weft: failed to specialize the entry argument\n";
    return 1;
  }

  // Build the pass pipeline: canonicalize, then barrier-thread-specialize.
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::barrier::createFullUnrollLoopsPass());
  pm.addPass(mlir::barrier::createBarrierThreadSpecializePass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "weft: pass pipeline failed\n";
    return 1;
  }

  module->print(llvm::outs());

  // Collect all thread programs from the input specialized module.
  std::vector<mlir::func::FuncOp> threadPrograms(numThreads);
  module->walk([&](mlir::func::FuncOp func) {
    mlir::IntegerAttr threadSpecializedAttr = llvm::dyn_cast<mlir::IntegerAttr>(
        func->getAttr(mlir::barrier::kThreadSpecializedAttrName));
    if (!threadSpecializedAttr)
      return;
    int tid = threadSpecializedAttr.getInt();
    threadPrograms[tid] = func;
  });
  for (int i = 0; i < numThreads; i++) {
    if (!threadPrograms[i]) {
      llvm::errs() << "weft: thread program " << i << " is not found\n";
      return 1;
    }
  }

  // Simulate the thread programs.
  std::map<mlir::Operation *, int> generations;
  llvm::LogicalResult simulationResult =
      simulateThreadPrograms(threadPrograms, generations);
  if (mlir::failed(simulationResult)) {
    llvm::errs() << "weft: failed to simulate the thread programs\n";
    return 1;
  }

  // Check if the program is well-synchronized.
  llvm::FailureOr<bool> wellSynchronizedResult =
      checkWellSynchronized(threadPrograms, generations);
  if (mlir::failed(wellSynchronizedResult)) {
    llvm::errs()
        << "weft: failed to check if the program is well-synchronized\n";
    return 1;
  }
  if (!wellSynchronizedResult.value()) {
    llvm::errs() << "weft: the program is not well-synchronized\n";
    return 1;
  }

  return 0;
}
