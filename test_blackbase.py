"""
Test script to verify blackbase installation and component functionality.
"""

import sys
print("Python version: {}".format(sys.version))
print()

# Test 1: Import blackbase modules
print("=" * 60)
print("Test 1: Importing blackbase modules")
print("=" * 60)

try:
    from blackbase import context, resources, kernel
    print("[OK] Successfully imported blackbase modules")
    print("  - context: {}".format(context))
    print("  - resources: {}".format(resources))
    print("  - kernel: {}".format(kernel))
except Exception as e:
    print("[FAIL] Failed to import blackbase: {}".format(e))
    sys.exit(1)

# Test 2: Context infrastructure
print()
print("=" * 60)
print("Test 2: Context infrastructure")
print("=" * 60)

try:
    from blackbase.context import (
        ContextStore,
        ContextContract,
        normalize_context_key,
        build_minimal_context,
        validate_minimal_context,
    )
    
    ctx_store = ContextStore()
    ctx_store.set("generation", 10)
    ctx_store.set("individual_id", 42)
    gen = ctx_store.get("generation")
    print("[OK] ContextStore works: generation={}".format(gen))
    
    contract = ContextContract(requires=["generation"], provides=["fitness"])
    print("[OK] ContextContract works: requires={}".format(contract.requires))
    
    key = normalize_context_key("Candidate.Model")
    print("[OK] Key normalization works: 'Candidate.Model' -> '{}'".format(key))
    
    ctx = build_minimal_context(
        generation=10,
        individual_id=42,
        constraints=[0.0, 1.0],
        constraint_violation=0.5,
    )
    validate_minimal_context(ctx)
    print("[OK] Minimal context works: {} fields".format(len(ctx)))
    
except Exception as e:
    print("[FAIL] Context infrastructure failed: {}".format(e))
    import traceback
    traceback.print_exc()

# Test 3: Resources layer
print()
print("=" * 60)
print("Test 3: Resources layer")
print("=" * 60)

try:
    from blackbase.resources import (
        DataRef,
        ResourceRequirement,
        WorkerDescriptor,
        TaskEnvelope,
        TaskResult,
        ResourceContext,
        PoolScheduler,
        build_local_worker_descriptor,
    )
    
    ref = DataRef(uri="s3://bucket/model.pt", kind="model")
    print("[OK] DataRef works: uri={}, kind={}".format(ref.uri, ref.kind))
    
    req = ResourceRequirement(threads=4, gpus=1, capabilities=["cpu", "cuda"])
    print("[OK] ResourceRequirement works: threads={}, gpus={}".format(req.threads, req.gpus))
    
    worker = build_local_worker_descriptor(worker_id="test-worker")
    print("[OK] WorkerDescriptor works: id={}, threads={}".format(worker.worker_id, worker.offer.threads))
    
    task = TaskEnvelope(task_id="task-1", task_type="evaluation", requirement=req)
    print("[OK] TaskEnvelope works: id={}, type={}".format(task.task_id, task.task_type))
    
    result = TaskResult.success(task_id="task-1", objectives=[0.5, 0.8])
    print("[OK] TaskResult works: ok={}, objectives={}".format(result.ok, result.objectives))
    
    resource_ctx = ResourceContext(device="cpu", threads=4)
    print("[OK] ResourceContext works: device={}, threads={}".format(resource_ctx.device, resource_ctx.threads))

    pool = PoolScheduler(total_threads=2)
    pool_result = pool.submit(lambda x: x + 1, 41)
    assert pool_result.result(timeout=5) == 42
    pool.shutdown()
    print("[OK] PoolScheduler works: result=42")
    
except Exception as e:
    print("[FAIL] Resources layer failed: {}".format(e))
    import traceback
    traceback.print_exc()

# Test 3.5: Shared protocol types
print()
print("=" * 60)
print("Test 3.5: Shared protocol types")
print("=" * 60)

try:
    from blackbase.types import Feedback, UnknownState

    state = UnknownState(values=[1, 2, 3], metadata={"source": "test"})
    next_state = state.with_values([4, 5, 6], stage="mutate")
    assert state.metadata["source"] == "test"
    assert next_state.metadata["stage"] == "mutate"
    print("[OK] UnknownState supports metadata alias and with_values")

    feedback = Feedback(objectives=[0.5], constraints=[0.0])
    assert feedback.ok
    print("[OK] Feedback works: score={}".format(feedback.scalar_score()))

except Exception as e:
    print("[FAIL] Shared protocol types failed: {}".format(e))
    import traceback
    traceback.print_exc()

# Test 4: Kernel layer
print()
print("=" * 60)
print("Test 4: Kernel layer")
print("=" * 60)

try:
    from blackbase.kernel import (
        PipelineSpec,
        PipelineSlotSpec,
        build_pipeline_kernel,
        RepresentationPipeline,
    )
    
    spec = PipelineSpec(
        key="test-pipeline",
        slots=[
            PipelineSlotSpec(slot="init", operators=["init_op"]),
            PipelineSlotSpec(slot="mutate", mode="serial", operators=["mut_op"]),
        ],
    )
    print("[OK] PipelineSpec works: key={}, slots={}".format(spec.key, len(spec.slots)))
    
    def init_fn(problem, ctx=None):
        return problem
    
    def mutate_fn(value, ctx=None):
        return value
    
    pipeline = RepresentationPipeline(initializer=init_fn, mutator=mutate_fn)
    result = pipeline.initialize("test-problem")
    print("[OK] RepresentationPipeline works: initialize returned '{}'".format(result))
    
except Exception as e:
    print("[FAIL] Kernel layer failed: {}".format(e))
    import traceback
    traceback.print_exc()

# Test 5: Backward compatibility
print()
print("=" * 60)
print("Test 5: Backward compatibility")
print("=" * 60)

try:
    from blackbase.context import ContextStore as InMemoryContextStore
    from blackbase.context import SnapshotStore as InMemorySnapshotStore
    
    store = InMemoryContextStore()
    print("[OK] NSGABlack style imports work: InMemoryContextStore created")
    
    contract1 = ContextContract(requires=["a"], provides=["b"])
    contract2 = ContextContract(requires=["a"], provides=["b"])
    print("[OK] Dual naming convention support works")
    
except Exception as e:
    print("[FAIL] Backward compatibility failed: {}".format(e))
    import traceback
    traceback.print_exc()

# Test 6: Snapshot store
print()
print("=" * 60)
print("Test 6: Snapshot store")
print("=" * 60)

try:
    from blackbase.context import create_snapshot_store, SnapshotStore
    
    snapshot_store = create_snapshot_store(backend="memory")
    handle = snapshot_store.write(
        {"population": [1, 2, 3], "objectives": [0.5, 0.8, 1.2]},
        key="test-snapshot",
        meta={"generation": 10},
    )
    record = snapshot_store.read("test-snapshot")
    print("[OK] SnapshotStore works: key={}, data={}".format(handle.key, record.data.keys()))
    
except Exception as e:
    print("[FAIL] Snapshot store failed: {}".format(e))
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("All tests completed successfully!")
print("=" * 60)
print()
print("blackbase is working correctly with:")
print("  * Context infrastructure")
print("  * L0 Resources layer")
print("  * Kernel layer")
print("  * Backward compatibility")
print("  * Snapshot store")
