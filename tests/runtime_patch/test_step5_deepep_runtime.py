# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for opt-in routing and asynchronous return lifetimes."""

import ast
import builtins
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


_env_spec = importlib.util.spec_from_file_location(
    "step5_deepep_test_envs", Path(__file__).parents[2] / "vllm_hcu/platforms/envs.py"
)
HENVS = importlib.util.module_from_spec(_env_spec)
_env_spec.loader.exec_module(HENVS)


def load_functions(*names):
    path = (
        Path(__file__).parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py"
    )
    nodes = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(nodes) == len(names)
    scope = dict(
        os=os,
        json=json,
        henvs=HENVS,
        _step5_ht_runtime_config=None,
        _step5_ht_runtime_applied=None,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def logger_modules():
    logger = ModuleType("vllm.logger")
    logger.init_logger = lambda _: SimpleNamespace(
        info=lambda *a, **k: None, info_once=lambda *a, **k: None
    )
    return {"vllm.logger": logger}


class RuntimeConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.scope = load_functions(
            "_apply_step5_ht_runtime_config", "step5_ht_launch_config"
        )

        class Buffer:
            num_sms = 18
            num_nvl_bytes = 1536 * 1024**2

            @classmethod
            def set_num_sms(cls, value):
                cls.num_sms = value

        self.prepare = SimpleNamespace(
            buffer=Buffer(),
            num_dispatchers_=16,
            _hcu_step5_runtime_tuning_allowed=True,
            _hcu_step5_buffer_pair=(object(), object()),
        )

    def apply(self, config):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "communication.json"
            path.write_text(json.dumps(config))
            with (
                patch.dict(os.environ, {"HCU_STEP5_DEEPEP_RUNTIME_CONFIG": str(path)}),
                patch.dict("sys.modules", logger_modules()),
            ):
                self.scope["_apply_step5_ht_runtime_config"](self.prepare)

    def test_non_step5_never_reads_profile_path(self):
        self.prepare._hcu_step5_runtime_tuning_allowed = False
        with patch.dict(os.environ, {"HCU_STEP5_DEEPEP_RUNTIME_CONFIG": "/missing"}):
            self.scope["_apply_step5_ht_runtime_config"](self.prepare)
        self.assertFalse(hasattr(self.prepare, "_hcu_step5_overlap"))

    def test_step5_profile_keeps_other_instances_and_library_defaults_unchanged(self):
        other = SimpleNamespace(
            buffer=type(self.prepare.buffer)(), _hcu_step5_runtime_tuning_allowed=False
        )
        original = object()
        self.apply(
            dict(
                version=1,
                num_sms=16,
                dispatch_send=8,
                combine_send=8,
                overlap=True,
                async_layout=True,
            )
        )
        self.assertEqual(type(other.buffer).num_sms, 18)
        self.assertIs(self.scope["step5_ht_launch_config"](other, original), original)
        self.assertFalse(hasattr(other, "_hcu_step5_num_sms"))

    def test_disabled_profile_returns_original_config_object(self):
        self.prepare._hcu_step5_runtime_tuning_allowed = False
        original = object()
        self.assertIs(
            self.scope["step5_ht_launch_config"](self.prepare, original), original
        )

    def test_runtime_budget_and_overlap_are_applied_together(self):
        self.apply(
            dict(
                version=1,
                num_sms=24,
                dispatch_send=8,
                combine_send=32,
                overlap=True,
                async_layout=True,
            )
        )
        self.assertEqual(self.prepare._hcu_step5_num_sms, 24)
        self.assertEqual(type(self.prepare.buffer).num_sms, 18)
        self.assertTrue(self.prepare._hcu_step5_overlap)
        self.assertTrue(self.prepare._hcu_step5_async_layout)
        self.assertEqual(self.prepare._hcu_step5_combine_send, 32)

    def test_overlap_cannot_use_single_shared_buffer(self):
        self.prepare._hcu_step5_buffer_pair = (object(),)
        with self.assertRaisesRegex(ValueError, "two preallocated"):
            self.apply(dict(version=1, num_sms=18, overlap=True))

    def test_mnvl_sm16_profile_is_accepted(self):
        self.apply(
            dict(
                version=1,
                num_sms=16,
                dispatch_send=8,
                combine_send=8,
                overlap=True,
                async_layout=True,
            )
        )
        self.assertEqual(self.prepare._hcu_step5_num_sms, 16)
        self.assertEqual(type(self.prepare.buffer).num_sms, 18)
        self.assertTrue(self.prepare._hcu_step5_overlap)

    def test_invalid_budget_is_rejected(self):
        for config in [
            dict(version=1, num_sms=6),
            dict(version=1, num_sms=16.0),
            dict(version=1, num_sms=16, dispatch_send=8.0),
            dict(version=1, num_sms=18, combine_send=128),
            dict(version=1, num_sms=18, overlap="true"),
            dict(version=1, num_sms=18, recv_tokens=64),
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.apply(config)

    def test_custom_geometry_cannot_exceed_reserved_buffer(self):
        class Config:
            def __init__(self, *args):
                pass

            def get_nvl_buffer_size_hint(self, *args):
                return self.required

        self.prepare._hcu_step5_combine_send = 32
        self.prepare._hcu_step5_num_sms = 16
        Config.required = self.prepare.buffer.num_nvl_bytes + 1
        with self.assertRaisesRegex(ValueError, "reserved NVL buffer"):
            self.scope["step5_ht_launch_config"](self.prepare, Config(), combine=True)
        Config.required = self.prepare.buffer.num_nvl_bytes
        self.assertIsInstance(
            self.scope["step5_ht_launch_config"](self.prepare, Config(), combine=True),
            Config,
        )


class ConstructionIsolationTest(unittest.TestCase):
    def setUp(self):
        self.configure = load_functions("configure_step5_ht_kernel")[
            "configure_step5_ht_kernel"
        ]

    def test_default_off_does_not_import_optional_runtime(self):
        kernel = object()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(self.configure(kernel, None, None), kernel)

    def modules(self):
        config = ModuleType("vllm.config")
        config.get_current_vllm_config_or_none = lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(enable_dbo=True)
        )
        step5 = ModuleType("vllm_hcu.runtime_compat.step5")
        step5.is_step5_owner = lambda owner: owner.step5
        return {"vllm.config": config, "vllm_hcu.runtime_compat.step5": step5}

    def test_other_models_ignore_enabled_step5_overlap_flags(self):
        prepare = type("DeepEPHTPrepareAndFinalize", (), {})()
        kernel = SimpleNamespace(
            prepare_finalize=prepare, fused_experts=SimpleNamespace()
        )
        with (
            patch.dict(
                os.environ,
                {
                    "HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT": "1",
                    "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR": "1",
                },
            ),
            patch.dict("sys.modules", self.modules()),
        ):
            self.assertIs(
                self.configure(kernel, None, SimpleNamespace(step5=False)), kernel
            )
        self.assertFalse(prepare._hcu_step5_runtime_tuning_allowed)
        self.assertFalse(prepare._hcu_step5_profile_dbo_output_requested)
        self.assertFalse(hasattr(prepare, "_hcu_step5_buffer_pair"))

    def test_unsupported_ep_group_rejected_before_optional_import_or_allocation(self):
        prepare = type("DeepEPHTPrepareAndFinalize", (), {})()
        prepare.num_dispatchers_ = 8
        kernel = SimpleNamespace(
            prepare_finalize=prepare, fused_experts=SimpleNamespace()
        )
        with (
            patch.dict(
                os.environ,
                {
                    "HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT": "1",
                    "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR": "1",
                    "DEEP_EP_NORMAL_MNVL": "1",
                },
            ),
            patch.dict("sys.modules", self.modules()),
        ):
            with self.assertRaisesRegex(ValueError, "EP16 NORMAL MNVL"):
                self.configure(kernel, None, SimpleNamespace(step5=True))
        self.assertFalse(hasattr(prepare, "_hcu_step5_buffer_pair"))


class FinalizeLifetimeTest(unittest.TestCase):
    def test_non_step5_finalizer_delegates_without_switching_streams(self):
        finalize = load_functions("ht_finalize_with_buffer_pair")[
            "ht_finalize_with_buffer_pair"
        ]
        module = SimpleNamespace(dbo_current_ubatch_id=lambda: 0)
        arguments = []

        def original(*args):
            arguments.append(args)
            return "legacy-receiver"

        values = [object() for _ in range(6)]
        owner = SimpleNamespace()
        result = finalize(module, owner, *values[:4], False, values[4], True, original)
        self.assertEqual(result, "legacy-receiver")
        self.assertIs(arguments[0][0], owner)
        self.assertIs(arguments[0][1], values[0])

    def test_return_waits_on_copy_stream_and_retains_inputs_until_receiver(self):
        finalize = load_functions("ht_finalize_with_buffer_pair")[
            "ht_finalize_with_buffer_pair"
        ]
        events = []
        dtype = object()

        class Tensor:
            is_cuda = True

            def __init__(self, name):
                self.name, self.dtype = name, dtype

            def numel(self):
                return 1

            def copy_(self, other, **kwargs):
                events.append(("copy", other.name))

            def record_stream(self, stream):
                events.append(("retain", self.name))

        source, output, returned = (
            Tensor("source"),
            Tensor("output"),
            Tensor("returned"),
        )

        class Delegate:
            pass

        class NoOp:
            def apply(self, **kwargs):
                return kwargs["fused_expert_output"]

        class Buffer:
            def capture(self):
                return object()

            def combine(self, **kwargs):
                self.arguments = kwargs
                events.append(("combine", kwargs["x"].name))
                event = SimpleNamespace(
                    event=object(),
                    current_stream_wait=lambda: events.append(
                        ("wait", "communication")
                    ),
                )
                return returned, None, event

        buffer = Buffer()
        owner = SimpleNamespace(
            _hcu_step5_pair_active={0: True},
            _hcu_step5_buffer_pair=(buffer, object()),
            _hcu_step5_inflight={0: [source]},
            handles=[object()],
            _get_combine_config=lambda: None,
        )
        module = SimpleNamespace(
            dbo_current_ubatch_id=lambda: 0,
            dbo_enabled=lambda: True,
            dbo_get_previous_event=lambda capture: capture(),
            dbo_yield_and_switch_from_compute_to_comm=lambda: events.append(
                ("yield", "communication")
            ),
            dbo_switch_to_compute=lambda: events.append(("switch", "compute")),
            dbo_switch_to_comm=lambda: events.append(("switch", "communication")),
            dbo_yield_and_switch_from_comm_to_compute=lambda: events.append(
                ("yield", "compute")
            ),
            TopKWeightAndReduceDelegate=Delegate,
            TopKWeightAndReduceContiguous=NoOp,
            torch=SimpleNamespace(
                Tensor=Tensor,
                bfloat16=dtype,
                cuda=SimpleNamespace(current_stream=lambda: "communication"),
            ),
        )
        receiver = finalize(
            module, owner, output, source, None, None, False, NoOp(), True, None
        )
        self.assertIn(0, owner._hcu_step5_inflight)
        self.assertNotIn(("copy", "returned"), events)
        receiver()
        self.assertLess(
            events.index(("switch", "communication")),
            events.index(("wait", "communication")),
        )
        self.assertLess(
            events.index(("wait", "communication")), events.index(("copy", "returned"))
        )
        self.assertIn(("retain", "source"), events)
        self.assertIn(("retain", "returned"), events)
        self.assertNotIn(0, owner._hcu_step5_inflight)
        self.assertFalse(owner._hcu_step5_pair_active[0])
        self.assertTrue(buffer.arguments["async_finish"])


class ProfileSelectionTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.scope = {
            "os": os,
            "re": re,
            "allows_static_step5_ll_eplb": lambda config: False,
        }
        path = root / "vllm_hcu/runtime_compat/step5.py"
        # Use the actual policy while substituting only the static EPLB gate.
        nodes = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
            and node.name != "allows_static_step5_ll_eplb"
        ]
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            self.scope,
        )
        config_module = ModuleType("vllm_hcu.patch.config")
        config_module.get_hcu_config = lambda config: config.additional_config
        modules = patch.dict("sys.modules", {config_module.__name__: config_module})
        modules.start()
        self.addCleanup(modules.stop)

    @staticmethod
    def model(backend="deepep_low_latency", *, model_type="step5", tp=4, dp=4):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type),
                quantization="slimquant_w4a8",
            ),
            parallel_config=SimpleNamespace(
                all2all_backend=backend,
                tensor_parallel_size=tp,
                data_parallel_size=dp,
                pipeline_parallel_size=1,
                enable_expert_parallel=True,
                enable_eplb=False,
            ),
            additional_config=None,
            lora_config=None,
        )

    @staticmethod
    def moe(backend="deepep_low_latency", *, owner=True, ep=16, dp=4):
        return SimpleNamespace(
            _hcu_step5_owner=owner,
            activation=SimpleNamespace(value="swiglustep"),
            moe_parallel_config=SimpleNamespace(
                all2all_backend=backend,
                ep_size=ep,
                dp_size=dp,
                enable_eplb=False,
                use_deepep_auto_kernels=False,
            ),
        )

    def test_foreign_models_ignore_inherited_step5_profiles(self):
        for phase in ("prefill", "decode", "auto", "invalid"):
            for model_type in ("qwen3", "deepseek_v3", "glm", "unknown"):
                with (
                    self.subTest(phase=phase, model=model_type),
                    patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": phase}),
                ):
                    config = self.model(model_type=model_type)
                    config.model_config.quantization = "unsupported"
                    self.assertIsNone(self.scope["profile_for_model"](config))
                    self.assertIsNone(
                        self.scope["resolve_profile"](self.moe(owner=False))
                    )

    def test_unset_profile_keeps_existing_model_and_moe_selection(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": ""}):
            self.assertIsNone(self.scope["profile_for_model"](self.model()))
            self.assertIsNone(self.scope["resolve_profile"](self.moe()))

    def test_foreign_models_do_not_read_step5_environment(self):
        def unexpected_read(*args):
            self.fail("foreign model read the Step5 profile")

        self.scope["os"] = SimpleNamespace(environ=SimpleNamespace(get=unexpected_read))
        self.assertIsNone(
            self.scope["profile_for_model"](self.model(model_type="qwen3"))
        )
        self.assertIsNone(self.scope["resolve_profile"](self.moe(owner=False)))

    def test_static_agrs_prefill_retains_its_topology_and_expert_count(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "prefill"}):
            config = self.model("allgather_reducescatter", tp=1, dp=8)
            moe = self.moe("allgather_reducescatter", ep=8, dp=8)
            self.assertEqual(self.scope["profile_for_model"](config), "prefill")
            self.assertEqual(self.scope["resolve_profile"](moe), "prefill")
            self.assertEqual(self.scope["static_agrs_local_experts"](moe), 44)
            config.parallel_config.tensor_parallel_size = 4
            with self.assertRaisesRegex(ValueError, "static TP/DP/PP"):
                self.scope["profile_for_model"](config)
            moe.moe_parallel_config.ep_size = 16
            with self.assertRaisesRegex(ValueError, "static AG/RS"):
                self.scope["resolve_profile"](moe)

    def test_static_agrs_decode_and_auto_fail_before_contiguous_selection(self):
        for phase in ("decode", "auto"):
            for dp, ep in ((2, 8), (4, 16)):
                with (
                    self.subTest(phase=phase, dp=dp),
                    patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": phase}),
                ):
                    with self.assertRaisesRegex(ValueError, "masked experts"):
                        self.scope["profile_for_model"](
                            self.model("allgather_reducescatter", tp=4, dp=dp)
                        )
                    with self.assertRaisesRegex(ValueError, "masked experts"):
                        self.scope["resolve_profile"](
                            self.moe("allgather_reducescatter", ep=ep, dp=dp)
                        )

    def test_fixed_ht_is_prefill_only(self):
        for phase in ("prefill", "decode", "auto"):
            with (
                self.subTest(phase=phase),
                patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": phase}),
            ):
                config = self.model("deepep_high_throughput")
                moe = self.moe("deepep_high_throughput")
                if phase == "prefill":
                    self.assertEqual(self.scope["profile_for_model"](config), phase)
                    self.assertEqual(self.scope["resolve_profile"](moe), "auto")
                else:
                    with self.assertRaisesRegex(ValueError, "masked experts"):
                        self.scope["profile_for_model"](config)
                    with self.assertRaisesRegex(ValueError, "masked experts"):
                        self.scope["resolve_profile"](moe)

    def test_deepep_low_latency_and_auto_preserve_framework_topologies(self):
        for phase in ("prefill", "decode", "auto"):
            for backend in ("deepep_low_latency", "deepep_auto"):
                for tp, dp in ((4, 2), (4, 4), (1, 16)):
                    with self.subTest(phase=phase, backend=backend, tp=tp, dp=dp):
                        with patch.dict(
                            os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": phase}
                        ):
                            self.assertEqual(
                                self.scope["profile_for_model"](
                                    self.model(backend, tp=tp, dp=dp)
                                ),
                                phase,
                            )
                            self.assertEqual(
                                self.scope["resolve_profile"](
                                    self.moe(backend, ep=tp * dp, dp=dp)
                                ),
                                "auto",
                            )

    def test_normalized_hcu_auto_flag_is_not_treated_as_fixed_ll(self):
        for phase in ("decode", "auto"):
            with (
                self.subTest(phase=phase),
                patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": phase}),
            ):
                config = self.model("deepep_low_latency")
                config.additional_config = SimpleNamespace(deepep_auto=True)
                moe = self.moe("deepep_low_latency")
                moe.moe_parallel_config.use_deepep_auto_kernels = True
                self.assertTrue(self.scope["uses_step5_deepep"](config))
                self.assertTrue(self.scope["uses_step5_deepep"](moe))
                self.assertEqual(self.scope["profile_for_model"](config), phase)
                self.assertEqual(self.scope["resolve_profile"](moe), "auto")

    def test_eplb_requires_the_existing_static_placement_predicate(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "decode"}):
            config, moe = self.model(), self.moe()
            config.parallel_config.enable_eplb = True
            moe.moe_parallel_config.enable_eplb = True
            for name, value in (
                ("profile_for_model", config),
                ("resolve_profile", moe),
            ):
                with self.subTest(selector=name):
                    with self.assertRaisesRegex(ValueError, "static expert placement"):
                        self.scope[name](value)
            self.scope["allows_static_step5_ll_eplb"] = lambda value: True
            self.assertEqual(self.scope["profile_for_model"](config), "decode")
            self.assertEqual(self.scope["resolve_profile"](moe), "auto")

    def test_quantization_lora_and_activation_contracts_remain_required(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "decode"}):
            config = self.model()
            config.model_config.quantization = "unsupported"
            with self.assertRaisesRegex(ValueError, "SlimQuant or channel GPTQ"):
                self.scope["profile_for_model"](config)
            config.model_config.quantization = "slimquant_w4a8"
            config.lora_config = object()
            with self.assertRaisesRegex(ValueError, "LoRA"):
                self.scope["profile_for_model"](config)
            moe = self.moe()
            moe.activation.value = "silu"
            self.assertIsNone(self.scope["resolve_profile"](moe))

    def test_prefill_padding_ignores_retired_decode_mode_switch(self):
        for mode in ("1002", "1006", "invalid"):
            with (
                self.subTest(mode=mode),
                patch.dict(
                    os.environ,
                    {
                        "VLLM_HCU_STEP5_W4A8_PROFILE": "prefill",
                        "HCU_STEP5_W4A8_MODE": mode,
                    },
                ),
            ):
                self.assertEqual(
                    self.scope["resolve_profile"](
                        self.moe("allgather_reducescatter", ep=8, dp=8)
                    ),
                    "prefill",
                )
                self.assertEqual(self.scope["padded_route_rows"](1, 8, 44), 512)


class GemmSelectionTest(unittest.TestCase):
    @staticmethod
    def load_function(
        filename,
        name,
        scope,
        directory="vllm_hcu/model_executor/layers/fused_moe",
    ):
        path = Path(__file__).resolve().parents[2] / directory / filename
        node = next(
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        node.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        return scope[name]

    class Tensor:
        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)
            self.views = []

        def view(self, *shape):
            self.views.append(shape)
            return type(self)(shape)

    def native_module(self):
        module = ModuleType("deepgemm")
        calls = []
        for name in (
            "m_grouped_w4a8_gemm_nt_masked_hipc",
            "m_grouped_w4a8_gemm_nt_contiguous_hipc",
            "m_grouped_fp8_gemm_nt_masked",
            "m_grouped_fp8_gemm_nt_masked_ll",
            "m_grouped_fp8_gemm_nt_contiguous",
        ):

            def record(*args, _name=name, **kwargs):
                calls.append((_name, args, kwargs))

            setattr(module, name, record)
        return module, calls

    def test_w4a8_decode_uses_masked_without_a_batch_or_mode_whitelist(self):
        gemm = self.load_function("step5_deepep.py", "_gemm", {})
        module, calls = self.native_module()
        for capacity in (16, 64, 272, 512):
            with (
                self.subTest(capacity=capacity),
                patch.dict("sys.modules", {"deepgemm": module}),
            ):
                weights = self.Tensor((22, 96, 32, 64))
                output = self.Tensor((22, capacity, 3072))
                gemm(
                    self.Tensor((22, capacity, 4096)),
                    weights,
                    output,
                    "activation_scales",
                    "channel_scales",
                    "device_counts",
                    17,
                    True,
                    True,
                )
                name, args, kwargs = calls[-1]
                self.assertEqual(name, "m_grouped_w4a8_gemm_nt_masked_hipc")
                self.assertEqual(args[0][1], "activation_scales")
                self.assertEqual(args[1][1], "channel_scales")
                self.assertEqual(args[-2:], ("device_counts", 17))
                self.assertIs(args[2], output)
                self.assertEqual(kwargs, {"config": None})
                self.assertEqual(weights.views, [(22, 3072, 2048)])

    def test_w4a8_prefill_retains_contiguous_mode_1002(self):
        gemm = self.load_function("step5_deepep.py", "_gemm", {})
        module, calls = self.native_module()
        weights = self.Tensor((22, 3072, 2048))
        with patch.dict("sys.modules", {"deepgemm": module}):
            gemm(
                self.Tensor((8192, 4096)),
                weights,
                self.Tensor((8192, 3072)),
                "activation_scales",
                "channel_scales",
                "expert_indices",
                0,
                True,
                False,
            )
        name, args, kwargs = calls[-1]
        self.assertEqual(name, "m_grouped_w4a8_gemm_nt_contiguous_hipc")
        self.assertEqual(args[-1], "expert_indices")
        self.assertEqual(kwargs, {"config": {"MODE": 1002}})
        self.assertEqual(weights.views, [])

    def test_contiguous_factory_rejects_decode_before_optional_imports(self):
        def unexpected_import(name, *args, **kwargs):
            if name == "__future__":
                return builtins.__import__(name, *args, **kwargs)
            self.fail("Decode loaded the contiguous experts' optional dependencies")

        scope = {"__builtins__": dict(vars(builtins), __import__=unexpected_import)}
        factory = self.load_function("step5_w4a8.py", "_make_experts_class", scope)
        with self.assertRaisesRegex(ValueError, "Prefill only"):
            factory("decode")

    def test_fp8_decode_is_masked_and_prefill_is_contiguous_for_every_mode(self):
        module, calls = self.native_module()
        for mode in (1000, 1001, 1101):
            with (
                self.subTest(mode=mode),
                patch.dict("sys.modules", {"deepgemm": module}),
            ):
                gemm = self.load_function(
                    "step5_deepep.py",
                    "grouped_fp8",
                    {"decode_mode": lambda: mode},
                )
                rank = 6 if mode == 1101 else 7
                gemm(
                    "activation",
                    (self.Tensor((1,) * rank), "scales"),
                    "output",
                    "device_counts",
                    17,
                    decode=True,
                )
                name, args, kwargs = calls[-1]
                self.assertEqual(
                    name,
                    "m_grouped_fp8_gemm_nt_masked_ll"
                    if mode == 1101
                    else "m_grouped_fp8_gemm_nt_masked",
                )
                self.assertEqual(args[-2:], ("device_counts", 17))
                self.assertEqual(kwargs, {"config": {"MODE": mode}})
                gemm(
                    "activation",
                    (self.Tensor((1, 1, 1)), "scales"),
                    "output",
                    "expert_indices",
                    0,
                    decode=False,
                )
                self.assertEqual(calls[-1][0], "m_grouped_fp8_gemm_nt_contiguous")
                self.assertEqual(calls[-1][2], {"config": {"MODE": 1000}})
                count = len(calls)
                with self.assertRaisesRegex(ValueError, f"rank-{rank}"):
                    gemm(
                        "activation",
                        (self.Tensor((1, 1, 1)), "scales"),
                        "output",
                        "device_counts",
                        17,
                        decode=True,
                    )
                self.assertEqual(len(calls), count)

    def test_ll_compatibility_opt_ins_do_not_import_native_stacks_by_default(self):
        def unexpected_import(name, *args, **kwargs):
            if name == "__future__":
                return builtins.__import__(name, *args, **kwargs)
            self.fail("disabled Step5 LL option imported its runtime")

        scope = {
            "os": os,
            "__builtins__": dict(vars(builtins), __import__=unexpected_import),
        }
        placement = self.load_function(
            "step5.py",
            "allows_static_step5_ll_eplb",
            scope,
            directory="vllm_hcu/runtime_compat",
        )
        install = self.load_function(
            "step5.py",
            "install_int64_eplb_mapping",
            scope,
            directory="vllm_hcu/runtime_compat",
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(placement(None))
            self.assertIsNone(install(object(), None))

    def test_ll_topology_preserves_supported_and_rejected_parallel_layouts(self):
        supports = self.load_function(
            "step5.py",
            "supports_step5_ll_topology",
            {},
            directory="vllm_hcu/runtime_compat",
        )
        for tp, dp in ((4, 2), (4, 4), (1, 16)):
            with self.subTest(tp=tp, dp=dp):
                self.assertTrue(supports(tp, dp, 1))
                self.assertFalse(supports(tp, dp, 2))
        for tp, dp in ((1, 1), (2, 8), (8, 2)):
            with self.subTest(tp=tp, dp=dp):
                self.assertFalse(supports(tp, dp, 1))

    def test_allocator_limit_is_default_off_and_bounds_the_device_fraction(self):
        apply_limit = self.load_function(
            "worker.py",
            "apply_allocator_limit",
            {"os": os, "math": math},
            directory="vllm_hcu/v1",
        )
        calls = []
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda device: SimpleNamespace(
                    total_memory=96 * 2**30
                ),
                set_per_process_memory_fraction=lambda fraction, device: calls.append(
                    (fraction, device)
                ),
            )
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(apply_limit("cuda:0"))
        for value, expected_fraction in (("48", 0.5), ("144", 0.98)):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"VLLM_HCU_ALLOCATOR_LIMIT_GIB": value}),
            ):
                self.assertEqual(
                    apply_limit("cuda:0", torch_module), int(value) * 2**30
                )
                self.assertEqual(calls[-1], (expected_fraction, "cuda:0"))
        for value in ("0", "-1", "nan", "inf", "invalid"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"VLLM_HCU_ALLOCATOR_LIMIT_GIB": value}),
            ):
                count = len(calls)
                with self.assertRaises(ValueError):
                    apply_limit("cuda:0", torch_module)
                self.assertEqual(len(calls), count)


class LoadedModelValidationTest(unittest.TestCase):
    def setUp(self):
        self.address = 0
        self.logs = []
        self.torch = SimpleNamespace(
            int8="int8", float32="fp32", bfloat16="bf16", float8_e4m3fn="fp8"
        )
        self.validate = GemmSelectionTest.load_function(
            "step5_w4a8.py",
            "validate_loaded_model",
            {
                "torch": self.torch,
                "logger": SimpleNamespace(info=lambda *args: self.logs.append(args)),
            },
        )

    def weight(self, shape):
        self.address += 1
        pointer = self.address
        storage = SimpleNamespace(
            data_ptr=lambda: pointer, nbytes=lambda: math.prod(shape)
        )
        weight = SimpleNamespace(
            shape=shape,
            dtype=self.torch.int8,
            device=SimpleNamespace(index=0),
            storage=storage,
        )
        weight.numel = lambda: math.prod(weight.shape)
        weight.element_size = lambda: 1
        weight.untyped_storage = lambda: weight.storage
        return weight

    def layer(
        self, *, declared=True, packed=True, deepep=True, kind=None, static=False
    ):
        layer = SimpleNamespace(
            w13_weight=self.weight((44, 3072, 2048) if static else (2, 4, 2)),
            w2_weight=self.weight((44, 4096, 768) if static else (2, 4, 2)),
        )
        if declared:
            layer._hcu_step5_w4a8_declared = True
        if packed:
            layer._hcu_step5_w4a8_owner = object()
        if deepep:
            layer._hcu_step5_deepep = {"backend": "deepep_low_latency"}
        if kind is not None:
            layer._hcu_step5_gptq_kind = kind
        return layer

    @staticmethod
    def model(layers):
        return SimpleNamespace(modules=lambda: iter(layers))

    @staticmethod
    def load_method(relative_path, class_name, method_name, scope):
        path = Path(__file__).resolve().parents[2] / relative_path
        owner = next(
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method = next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        return scope[method_name]

    def test_slimquant_auto_and_fixed_ll_use_all_local_w4_declarations(self):
        for phase in ("auto", "decode"):
            with (
                self.subTest(phase=phase),
                patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": phase}),
            ):
                layers = [self.layer(), self.layer()]
                self.assertFalse(
                    any(hasattr(m, "_hcu_step5_gptq_kind") for m in layers)
                )
                self.validate(self.model(layers))
                self.assertEqual(self.logs[-1][1:3], (2, 0))

    def test_gptq_mixed_and_shared_declarations_are_not_double_counted(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "decode"}):
            layers = [
                self.layer(declared=False, kind="w4a8"),
                self.layer(kind="w4a8"),
                self.layer(declared=False, packed=False, kind="fp8"),
                self.layer(declared=False, packed=False, deepep=False, kind="bf16"),
            ]
            self.validate(self.model(layers))
            self.assertEqual(self.logs[-1][1:3], (2, 2))

    def test_declared_w4_layer_without_a_packed_owner_is_rejected(self):
        for phase in ("auto", "decode"):
            for kind in (None, "w4a8"):
                with (
                    self.subTest(phase=phase, kind=kind),
                    patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": phase}),
                ):
                    layers = [
                        self.layer(kind=kind),
                        self.layer(packed=False, kind=kind),
                    ]
                    with self.assertRaisesRegex(RuntimeError, "one packed owner"):
                        self.validate(self.model(layers))

    def test_an_undeclared_packed_owner_is_rejected(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "decode"}):
            with self.assertRaisesRegex(RuntimeError, "one packed owner"):
                self.validate(self.model([self.layer(), self.layer(declared=False)]))

    def test_aliasing_w4_storage_is_rejected_within_and_across_layers(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "decode"}):
            for across_layers in (False, True):
                with self.subTest(across_layers=across_layers):
                    layers = [self.layer(), self.layer()]
                    target = (
                        layers[1].w13_weight if across_layers else layers[0].w2_weight
                    )
                    target.storage = layers[0].w13_weight.storage
                    with self.assertRaisesRegex(RuntimeError, "one packed owner"):
                        self.validate(self.model(layers))

    def test_extra_backing_storage_cannot_hide_behind_a_single_owner(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "decode"}):
            layer = self.layer()
            layer.w13_weight.storage.nbytes = lambda: 2 * layer.w13_weight.numel()
            with self.assertRaisesRegex(RuntimeError, "one packed owner"):
                self.validate(self.model([layer]))

    def test_static_prefill_still_requires_all_88_layers_and_original_geometry(self):
        with patch.dict(os.environ, {"VLLM_HCU_STEP5_W4A8_PROFILE": "prefill"}):
            layers = [self.layer(deepep=False, static=True) for _ in range(88)]
            self.validate(self.model(layers))
            self.assertEqual(self.logs[-1][1:4], (88, 0, 0))
            with self.assertRaisesRegex(RuntimeError, "one packed owner"):
                self.validate(self.model(layers[:-1]))
            layers[0].w13_weight.shape = (44, 3071, 2048)
            with self.assertRaisesRegex(RuntimeError, "one packed owner"):
                self.validate(self.model(layers))

    def test_create_weights_declares_only_resolved_step5_profiles(self):
        self.torch.empty = self.torch.ones = lambda *shape, **kwargs: self.weight(shape)
        self.torch.nn = SimpleNamespace(Parameter=lambda tensor, **kwargs: tensor)
        scope = {
            "torch": self.torch,
            "set_weight_attrs": lambda *args: None,
            "FusedMoeWeightScaleSupported": SimpleNamespace(
                CHANNEL=SimpleNamespace(value="channel")
            ),
        }
        create_weights = self.load_method(
            "vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py",
            "SlimQuantW4A8Int8AiterMoEMethod",
            "create_weights",
            scope,
        )
        for profile in (None, "prefill", "decode", "auto"):
            with self.subTest(profile=profile):
                layer = SimpleNamespace()
                layer.register_parameter = lambda name, value: setattr(
                    layer, name, value
                )
                create_weights(
                    SimpleNamespace(_step5_w4a8_profile=profile), layer, 2, 4, 2, "bf16"
                )
                self.assertEqual(
                    getattr(layer, "_hcu_step5_w4a8_declared", False),
                    profile is not None,
                )

    def test_bf16_expert_conversion_obeys_master_and_preserves_source_delegation(self):
        class RoutedExperts:
            pass

        delegated = []
        native_method = object()
        source = SimpleNamespace(
            get_scheme_dict=lambda *args: None,
            get_quant_method=lambda *args: (delegated.append(args), native_method)[1],
        )
        select = self.load_method(
            "vllm_hcu/model_executor/layers/quantization/step5_gptq_w4a8.py",
            "Step5GPTQW4A8Config",
            "get_quant_method",
            {
                "os": os,
                "henvs": HENVS,
                "RoutedExperts": RoutedExperts,
                "UnquantizedFusedMoEMethod": lambda config: "bf16",
                "Step5BF16ToFP8MoEMethod": lambda config, prefix: "fp8",
            },
        )
        config = SimpleNamespace(source=source)
        prefix = "model.layers.0.moe.experts.routed_experts"
        for master, child, expected in (
            ("0", "1", "bf16"),
            ("1", "1", "fp8"),
            ("1", "0", "bf16"),
            ("0", "0", "bf16"),
        ):
            with (
                self.subTest(master=master, child=child),
                patch.dict(
                    os.environ,
                    {
                        "VLLM_HCU_USE_CUSTOM_OPS": master,
                        "VLLM_HCU_STEP5_FP8_BF16_MOE": child,
                    },
                ),
            ):
                layer = RoutedExperts()
                layer.moe_config = object()
                self.assertEqual(select(config, layer, prefix), expected)
                self.assertEqual(layer._hcu_step5_gptq_kind, expected)
        unrelated = object()
        self.assertIs(select(config, unrelated, prefix), native_method)
        self.assertEqual(delegated, [(unrelated, prefix)])


if __name__ == "__main__":
    unittest.main()
