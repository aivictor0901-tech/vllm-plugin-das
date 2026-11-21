"""Foreign models and unset recipes must never enter Step5-specific paths."""

import ast
import builtins
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    fn.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


def config(model="qwen3"):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)),
        parallel_config=SimpleNamespace(
            data_parallel_size=4, tensor_parallel_size=4, enable_dbo=False
        ),
    )


def step5(c):
    return c is not None and c.model_config.hf_config.model_type == "step5"


@pytest.mark.parametrize("model", ["qwen3", "deepseek_v3", "glm", "unknown"])
def test_foreign_worker_ignores_inherited_dbo_flag(model):
    fn = function(
        "vllm_hcu/v1/hcu_dbo.py",
        "enable_in_worker",
        dict(
            is_step5_config=step5,
            DBO_MIN_TOKENS=4096,
            henvs=SimpleNamespace(optional_custom_op_enabled=lambda x: x),
        ),
    )
    c = config(model)
    assert fn(c) is False and not c.parallel_config.enable_dbo


@pytest.mark.parametrize(
    "model,threshold", [("qwen3", 4096), ("qwen3", 0), ("step5", 0)]
)
def test_native_ubatch_runner_is_not_wrapped(model, threshold):
    ur = SimpleNamespace(prepare=object(), run=object())
    original = dict(vars(ur))
    fn = function(
        "vllm_hcu/v1/hcu_dbo.py",
        "install_runner",
        dict(
            DBO_MIN_TOKENS=threshold,
            is_step5_config=step5,
            henvs=SimpleNamespace(optional_custom_op_enabled=lambda x: x),
        ),
    )
    fn(SimpleNamespace(vllm_config=config(model), ubatch_runner=ur))
    assert vars(ur) == original


@pytest.mark.parametrize("owner", [None, config(), config("glm")])
def test_fabric_rejects_foreign_or_unknown_owner_before_device_probe(owner):
    fn = function(
        "vllm_hcu/distributed/device_communicators/step5_dp_fabric.py",
        "maybe_install",
        dict(
            os=SimpleNamespace(environ={"HCU_STEP5_DP_FABRIC": "1"}),
            is_step5_config=step5,
            henvs=SimpleNamespace(optional_custom_op_enabled=lambda x: x),
        ),
    )
    native = object()
    comm = SimpleNamespace(world_size=4, pynccl_comm=native)
    fn(comm, None, None, "dp:test", vllm_config=owner)
    assert comm.pynccl_comm is native


def test_other_gpu_never_reads_step5_moe_table():
    fn = function(
        "vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py",
        "pick_configs",
        dict(_device_arch=lambda _: "gfx942", lib_configs=lambda *a: "native"),
    )
    assert (
        fn(
            SimpleNamespace(shape=(22, 3072, 4096), device=SimpleNamespace(index=0)),
            object(),
            8,
            65536,
        )
        == "native"
    )


def load_tuning():
    spec = importlib.util.spec_from_file_location(
        "step5_tuning_subject", ROOT / "vllm_hcu/ops/step5/tuning.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_recipe_has_no_dense_or_asm_choices(monkeypatch):
    monkeypatch.delenv("HCU_STEP5_TUNING_CONFIG", raising=False)
    t = load_tuning()
    assert t.dense_recipe() is None and t.asm_recipe(22) is None


def test_external_recipe_is_data_and_does_not_execute_code(monkeypatch, tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=1,
                architecture="gfx938",
                dense=dict(hipblaslt_sha256="a" * 64, algorithms={"8,16,32": 7}),
                asm_g1=dict(
                    block_m=128, solution=7, persist_by_local_experts={"22": 16}
                ),
            )
        )
    )
    monkeypatch.setenv("HCU_STEP5_TUNING_CONFIG", str(path))
    t = load_tuning()
    assert t.dense_recipe() == ({(8, 16, 32): 7}, "a" * 64)
    assert t.asm_recipe(22) == (7, 16)
    path.write_text("__import__('os').abort()")
    with pytest.raises(json.JSONDecodeError):
        load_tuning().tuning_data()


@pytest.mark.parametrize("build", [False, True])
def test_dense_build_dependency_is_explicit(monkeypatch, build):
    monkeypatch.setenv("VLLM_HCU_BUILD_STEP5_DENSE", str(int(build)))
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in {"_BUILD_STEP5_DENSE", "ext_modules"}
            for t in n.targets
        )
    ]
    ns = dict(os=os, CUDAExtension=lambda **kw: kw)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "setup.py", "exec"), ns)
    extension = ns["ext_modules"][0]
    assert ("hipblaslt" in extension["libraries"]) is build
    assert ("vllm_hcu/csrc/step5_fp8_dense.cpp" in extension["sources"]) is build
    assert extension["extra_compile_args"]["nvcc"] == ["-O3", "-fno-gpu-rdc"]


@pytest.mark.parametrize("step5_layer", [False, True])
def test_w8a8_dispatch_uses_step5_path_only_for_step5_linears(step5_layer):
    fn = function(
        "vllm_hcu/patch/worker/op_opt/patch_compressed_tensors_w8a8_int8.py",
        "hcu_apply_weights",
        dict(
            _custom_quantization_enabled=lambda: True,
            apply_int8_linear=lambda **kwargs: ("legacy", kwargs),
            apply_step5_int8_linear=lambda **kwargs: ("step5", kwargs),
        ),
    )
    layer = SimpleNamespace(weight=object(), weight_scale=object(), params_dtype=object())
    if step5_layer:
        layer._hcu_step5_int8_linear = True
    pair = (object(), object())
    route, args = fn(None, layer, object(), None, pair)
    assert route == ("step5" if step5_layer else "legacy")
    assert args["x_and_scale_quanted"] is pair


def test_step5_int8_owner_covers_mtp_head_without_tagging_non_linear_layers():
    class Linear:
        pass

    class MtpHead:
        pass

    mark = function(
        "vllm_hcu/models/step5_text.py",
        "_mark_step5_int8_linears",
        dict(LinearBase=Linear, ParallelLMHead=MtpHead),
    )
    linear, head, other = Linear(), MtpHead(), SimpleNamespace()
    mark(SimpleNamespace(modules=lambda: (linear, head, other)))
    assert linear._hcu_step5_int8_linear and head._hcu_step5_int8_linear
    assert not hasattr(other, "_hcu_step5_int8_linear")


def test_retired_runtime_and_tuning_artifacts_are_absent():
    for name in [
        "vllm_hcu/ops/step5/prefill_g1.py",
        "vllm_hcu/distributed/static_eplb_diagnostics.py",
        "vllm_hcu/model_executor/layers/fused_moe/ep_table.json",
    ]:
        assert not (ROOT / name).exists()


def test_threshold_hook_uses_parallel_owner_marker_and_keeps_native_dbo():
    fn = function(
        "vllm_hcu/v1/hcu_dbo.py",
        "check_ubatch_thresholds",
        dict(
            original_threshold=lambda *args: "native",
            DBO_MIN_TOKENS=4096,
            _MIN_TAIL=64,
            sys=SimpleNamespace(_getframe=lambda _: SimpleNamespace(f_locals={})),
        ),
    )
    assert fn(SimpleNamespace(), 8192, False) == "native"
    # A count alone no longer establishes safe query boundaries or DP agreement.
    assert fn(SimpleNamespace(_hcu_step5_dbo=True), 8192, False) is False
    assert fn(SimpleNamespace(_hcu_step5_dbo=True), 8192, True) is False


def test_foreign_shared_experts_ignore_inherited_late_stream_threshold(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    spec = importlib.util.spec_from_file_location(
        "step5_late_stream_test_envs", ROOT / "vllm_hcu/platforms/envs.py"
    )
    henvs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(henvs)
    fn = function(
        "vllm_hcu/model_executor/layers/fused_moe/shared_experts.py",
        "_late_stream",
        dict(henvs=henvs, _LATE_STREAM_MIN_TOKENS=1, dbo_enabled=lambda: False),
    )
    hidden = SimpleNamespace(shape=(65536, 4096))
    assert (
        fn(SimpleNamespace(_hcu_step5_owner=False, _stream=object()), hidden) is False
    )
    assert fn(SimpleNamespace(_hcu_step5_owner=True, _stream=object()), hidden) is True


@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("owns_dispatch", [False, True])
def test_combine_hooks_install_only_for_step5_constructor(owner, owns_dispatch):
    path = ROOT / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoERunner"
    )
    ctor = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    index = next(
        i
        for i, n in enumerate(ctor.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_ep_fused_combine"
            for t in n.targets
        )
    )
    calls = []
    ns = dict(
        self=SimpleNamespace(
            _hcu_step5_owner=owner,
            _quant_method=SimpleNamespace(hcu_owns_dispatch=owns_dispatch),
        ),
        _EP_FUSED_COMBINE=True,
        shared_experts=object(),
        moe_config=SimpleNamespace(
            dp_size=4,
            pcp_size=1,
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
        ),
        routed_scaling_factor=1.0,
        routed_input_transform=None,
        routed_output_transform=None,
        _install_step5_combine_hooks=lambda: calls.append("installed"),
    )
    exec(
        compile(
            ast.Module(body=ctor.body[index : index + 2], type_ignores=[]),
            str(path),
            "exec",
        ),
        ns,
    )
    assert calls == (["installed"] if owner and not owns_dispatch else [])
    assert not any(
        isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "_EP_FUSED_COMBINE"
        for n in tree.body
    )


@pytest.fixture
def step5_decode_policy(monkeypatch):
    namespace = {
        "os": os,
        "allows_static_step5_ll_eplb": lambda config: False,
    }
    for name in ("is_step5_config", "is_step5_owner"):
        function("vllm_hcu/runtime_compat/step5.py", name, namespace)
    for name in ("uses_step5_deepep", "profile_for_model", "resolve_profile"):
        function(
            "vllm_hcu/runtime_compat/step5.py",
            name,
            namespace,
        )
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.patch.config",
        SimpleNamespace(
            get_hcu_config=lambda config: SimpleNamespace(deepep_auto=False)
        ),
    )
    return namespace


def step5_collective_configs(backend, tp, dp, enable_ep=True):
    model = config("step5")
    model.model_config.quantization = "slimquant_w4a8"
    model.parallel_config = SimpleNamespace(
        all2all_backend=backend,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        pipeline_parallel_size=1,
        enable_expert_parallel=enable_ep,
        enable_eplb=False,
    )
    model.additional_config = None
    model.lora_config = None
    moe = SimpleNamespace(
        _hcu_step5_owner=True,
        activation=SimpleNamespace(value="swiglustep"),
        moe_parallel_config=SimpleNamespace(
            all2all_backend=backend,
            use_deepep_auto_kernels=False,
            use_all2all_kernels=enable_ep and tp * dp > 1,
            enable_eplb=False,
        ),
    )
    return model, moe


@pytest.mark.parametrize("phase", ["decode", "auto"])
@pytest.mark.parametrize("backend", ["deepep_low_latency", "deepep_auto"])
@pytest.mark.parametrize(
    "tp,dp,enable_ep", [(1, 1, True), (4, 1, False), (4, 4, False)]
)
def test_decode_rejects_local_ep1_and_ordinary_tp_before_model_load(
    monkeypatch, step5_decode_policy, phase, backend, tp, dp, enable_ep
):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    model, moe = step5_collective_configs(backend, tp, dp, enable_ep)
    for selector, value in (("profile_for_model", model), ("resolve_profile", moe)):
        with pytest.raises(ValueError, match="distributed EP.*masked experts"):
            step5_decode_policy[selector](value)


@pytest.mark.parametrize("phase", ["decode", "auto"])
@pytest.mark.parametrize("backend", ["deepep_low_latency", "deepep_auto"])
def test_decode_uses_actual_all2all_capability_not_only_backend_name(
    monkeypatch, step5_decode_policy, phase, backend
):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    _, moe = step5_collective_configs(backend, 4, 4)
    moe.moe_parallel_config.use_all2all_kernels = False
    with pytest.raises(ValueError, match="distributed EP.*masked experts"):
        step5_decode_policy["resolve_profile"](moe)


@pytest.mark.parametrize("phase", ["decode", "auto"])
@pytest.mark.parametrize("backend", ["deepep_low_latency", "deepep_auto"])
@pytest.mark.parametrize("tp,dp", [(4, 2), (4, 4), (1, 16)])
def test_decode_keeps_valid_ep8_and_ep16_collective_profiles(
    monkeypatch, step5_decode_policy, phase, backend, tp, dp
):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    model, moe = step5_collective_configs(backend, tp, dp)
    assert step5_decode_policy["profile_for_model"](model) == phase
    assert step5_decode_policy["resolve_profile"](moe) == "auto"


@pytest.mark.parametrize(
    "tp,dp,enable_ep", [(1, 1, True), (4, 1, False), (4, 4, False)]
)
def test_prefill_keeps_existing_local_deepgemm_ownership(
    monkeypatch, step5_decode_policy, tp, dp, enable_ep
):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", "prefill")
    model, moe = step5_collective_configs("deepep_low_latency", tp, dp, enable_ep)
    assert step5_decode_policy["profile_for_model"](model) == "prefill"
    assert step5_decode_policy["resolve_profile"](moe) == "auto"


def test_foreign_local_models_ignore_inherited_decode_profile(
    monkeypatch, step5_decode_policy
):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", "decode")
    model, moe = step5_collective_configs("deepep_low_latency", 1, 1, False)
    model.model_config.hf_config.model_type = "qwen3"
    moe._hcu_step5_owner = False
    assert step5_decode_policy["profile_for_model"](model) is None
    assert step5_decode_policy["resolve_profile"](moe) is None


@pytest.mark.parametrize("phase", ["decode", "auto"])
@pytest.mark.parametrize(
    "backend,collective",
    [("deepep_low_latency", False), ("deepep_high_throughput", True)],
)
def test_prepare_rejects_contiguous_decode_before_optional_imports(
    monkeypatch, phase, backend, collective
):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)

    def guarded_import(name, *args, **kwargs):
        if name == "__future__":
            return builtins.__import__(name, *args, **kwargs)
        pytest.fail(
            "invalid Decode prepared optional runtime before rejecting its topology"
        )

    prepare = function(
        "vllm_hcu/model_executor/layers/fused_moe/step5_deepep.py",
        "prepare_step5_deepep",
        {"os": os, "__builtins__": dict(vars(builtins), __import__=guarded_import)},
    )
    _, moe = step5_collective_configs(backend, 4, 4)
    moe.moe_parallel_config.use_all2all_kernels = collective
    with pytest.raises(ValueError, match="distributed EP.*masked experts"):
        prepare(SimpleNamespace(moe=moe), None, True)
