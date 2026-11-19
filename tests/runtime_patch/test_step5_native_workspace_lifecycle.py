"""CPU lifetime/lock and real logits-entry ordering, without Torch or a GPU."""
import ast
import gc
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[2] / 'vllm_hcu'
NATIVE = ROOT / 'model_executor/layers/quantization/ct_w4a8_step5_native.py'


def workspace_api():
    class Owner:
        def __init__(self, count):
            self.count = count
        def numel(self):
            return self.count
    state = dict(device=0, streams={0: 7, 1: 9}, allocations=[], alive_at_allocation=[])
    def device(value, index=None):
        if isinstance(value, NS):
            return value
        kind, _, suffix = value.partition(':')
        return NS(type=kind, index=index if index is not None else int(suffix) if suffix else None)
    def empty(count, **kwargs):
        state['alive_at_allocation'].append(sum(ref() is not None for ref in state['allocations']))
        value = Owner(count)
        state['allocations'].append(weakref.ref(value))
        return value
    def forbidden(*args, **kwargs):
        raise AssertionError('No synchronization or allocator cache flush allowed')
    torch = NS(int8='int8', bfloat16='bf16', device=device, empty=empty,
        cuda=NS(current_device=lambda: state['device'],
            current_stream=lambda d: NS(cuda_stream=state['streams'][state['device'] if d.index is None else d.index]),
            synchronize=forbidden, empty_cache=forbidden))
    tree = ast.parse(NATIVE.read_text())
    names = {'_Workspace', '_workspace', 'native_workspace_inventory', 'release_native_workspaces'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    namespace = dict(torch=torch, threading=threading, _WORKSPACE_LOCK=threading.RLock(),
        _WORKSPACES={}, workspace_bytes=lambda rows, h, i: dict(INT8=rows * h, BF16=2 * rows * h))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-workspace-lifecycle>', 'exec'), namespace)
    return namespace, state, device


def logits_function(file, class_name, function_name):
    tree = ast.parse((ROOT / 'models' / file).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, '<actual-logits-entry>', 'exec'), namespace)
    return namespace[function_name]


class WorkspaceLifecycle(unittest.TestCase):
    def test_small_large_small_growth_drops_old_owner_before_new_allocation(self):
        api, state, device = workspace_api()
        x = NS(device=device('cuda:0'))
        pool = api['_workspace'](x)
        pool.reserve(13, 4096, 1536, x.device)
        old = [weakref.ref(pool.integer), weakref.ref(pool.bf16)]
        pool.reserve(29, 4096, 1536, x.device)
        self.assertTrue(all(ref() is None for ref in old))
        # Only the other buffer owner may survive during either allocation;
        # keeping the replaced owner alive would produce two instead of one.
        self.assertEqual(state['alive_at_allocation'], [0, 1, 1, 1])
        larger = (pool.integer, pool.bf16)
        pool.reserve(1, 4096, 1536, x.device)
        self.assertEqual((pool.integer, pool.bf16), larger)
        self.assertEqual(len(state['allocations']), 4)

    def test_current_stream_owners_released_idempotently_then_same_pool_reserves_again(self):
        api, state, device = workspace_api()
        x = NS(device=device('cuda:0'))
        pool = api['_workspace'](x)
        pool.reserve(13, 4096, 1536, x.device)
        owners = [weakref.ref(pool.integer), weakref.ref(pool.bf16)]
        self.assertEqual(api['release_native_workspaces'](x.device), 13 * 4096 * 3)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in owners))
        self.assertEqual(api['release_native_workspaces'](), 0)
        self.assertIs(api['_workspace'](x), pool)
        self.assertEqual(api['native_workspace_inventory'](), [
            dict(device=0, stream=7, INT8_bytes=0, BF16_bytes=0)])
        pool.reserve(13, 4096, 1536, x.device)
        self.assertIsNotNone(pool.integer)
        self.assertIsNotNone(pool.bf16)
        self.assertEqual(api['native_workspace_inventory']()[0]['BF16_bytes'], 13 * 4096 * 2)

    def test_other_stream_and_other_device_owners_are_not_freed(self):
        api, state, device = workspace_api()
        x = NS(device=device('cuda:0'))
        first = api['_workspace'](x)
        first.reserve(1, 4096, 1536, x.device)
        state['streams'][0] = 11
        other = api['_workspace'](x)
        other.reserve(2, 4096, 1536, x.device)
        state['device'] = 1
        remote = api['_workspace'](NS(device=device('cuda:1')))
        remote.reserve(3, 4096, 1536, device('cuda:1'))
        self.assertEqual(api['release_native_workspaces'](device('cuda:0')), 2 * 4096 * 3)
        self.assertIsNotNone(first.integer)
        self.assertIsNotNone(remote.integer)
        self.assertIsNone(other.integer)
        with self.assertRaises(ValueError):
            api['release_native_workspaces'](device('cpu'))

    def test_release_waits_for_enqueue_lock_without_holding_registry_lock(self):
        api, _, device = workspace_api()
        pool = api['_workspace'](NS(device=device('cuda:0')))
        pool.reserve(2, 4096, 1536, device('cuda:0'))
        entered, done, released = threading.Event(), threading.Event(), []
        def release():
            entered.set()
            released.append(api['release_native_workspaces'](device('cuda:0')))
            done.set()
        pool.lock.acquire()
        worker = threading.Thread(target=release)
        try:
            worker.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(done.wait(.03))
            self.assertIsNotNone(pool.integer)
            acquired = api['_WORKSPACE_LOCK'].acquire(timeout=.2)
            self.assertTrue(acquired, 'Registry lock must not be held while waiting for pool lock')
            if acquired:
                api['_WORKSPACE_LOCK'].release()
        finally:
            pool.lock.release()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(done.is_set())
        self.assertEqual(released, [2 * 4096 * 3])

    def test_main_and_real_mtp_head_entries_release_before_every_allocation_only_when_enabled(self):
        main = logits_function('step5_text.py', 'Step5ForCausalLM', 'compute_logits')
        inner = logits_function('step5_mtp.py', 'Step5MultiTokenPredictor', 'compute_logits')
        top = logits_function('step5_mtp.py', 'Step5MultiTokenPredictor', 'get_top_tokens')
        outer = logits_function('step5_mtp.py', 'Step5MTP', 'compute_logits')
        policy_name = 'vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5_native_policy'
        native_name = 'vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5_native'
        for enabled in (False, True):
            events = []
            policy, native = ModuleType(policy_name), ModuleType(native_name)
            policy.native_enabled = lambda: enabled
            native.release_native_workspaces = lambda device: events.append(('release', device))
            class Shared:
                head = 'mtp-head'
                def __call__(self, hidden):
                    events.append(('shared-head-normalize',))
                    return hidden
            class Processor:
                def __call__(self, head, hidden):
                    events.append(('allocate-logits', head))
                    return 'logits'
                def get_top_tokens(self, head, hidden):
                    events.append(('allocate-top-tokens', head))
                    return 'top-tokens'
            x = NS(device='cuda:0')
            def norm(hidden):
                events.append(('target-normalize',))
                return hidden
            shared = Shared()
            target = NS(model=NS(norm=norm), lm_head='target-head', logits_processor=Processor())
            def layer(index):
                events.append(('select-mtp-layer', index))
                return NS(shared_head=shared)
            mtp = NS(_layer=layer, logits_processor=Processor())
            outer_owner = NS(model=NS(compute_logits=lambda hidden, index: inner(mtp, hidden, index)))
            with patch.dict(sys.modules, {policy_name: policy, native_name: native}):
                for function, owner, expected in [(main, target, 'logits'),
                        (inner, mtp, 'logits'), (top, mtp, 'top-tokens'), (outer, outer_owner, 'logits')]:
                    # The exact same real entry is used in profiling and live
                    # inference: no profile/dummy discriminator exists here.
                    for repeat in range(2):
                        events.clear()
                        self.assertEqual(function(owner, x), expected)
                        releases = [e for e in events if e[0] == 'release']
                        self.assertEqual(len(releases), int(enabled))
                        if enabled:
                            self.assertEqual(events[0], ('release', 'cuda:0'))
                        self.assertTrue(events[-1][0].startswith('allocate-'))


if __name__ == '__main__':
    unittest.main()
