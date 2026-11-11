# Step5 W4A8 DeepEP + DeepGEMM

`serve_gptq_deepep.sh` selects DeepEP HT for prefill/mixed batches and LL for
pure decode, using the scheduler's globally synchronized phase. It keeps the
existing AG/RS launch recipes available separately.

Set `ALL2ALL_BACKEND=deepep_low_latency` to pin a Decode service to LL, or
`deepep_high_throughput` to pin HT. Fixed LL supports the mixed FP8 layers too;
the `STEP5_TEST_BACKEND` variable selects these modes in the block probe.

```bash
MODEL_PATH=/model/step-5-preview-channel-w4a8-GPTQ \
TP=4 DP=2 PP=1 CHUNK_SIZE=8192 \
bash benchmarks/step5_w4a8_pd/serve_gptq_deepep.sh --dry-run

# Remove --dry-run to serve. The same entry supports TP=1 DP=8.
```

TP/DP/EP/PP groups and local expert ownership come from vLLM, without an EP8
or 44-local-expert assertion. Kernel hidden-size/alignment support, expert
divisibility, memory capacity and the framework's executor restrictions still
apply. Dynamic EPLB and LoRA are not enabled. PP must use a supported executor;
Model Runner V2 rejects PP with `external_launcher`.

## Data and storage ownership

- Each DP batch is split across its TP ranks before routing, even when compiler
  sequence parallelism is disabled. A token enters DeepEP on one sender.
- W4A8 HT dispatch sends INT8 activations and FP32 token scales. Fused scatter
  writes aligned expert-major rows directly. `HCU_STEP5_DEEPEP_FUSED_SCATTER=0`
  selects ordinary dispatch followed by permutation.
- W4A8 LL calls the native BF16-input API with `quant_type=1`. Native dispatch
  quantizes the activations to INT8; an external prequantized-INT8 LL input is
  not supported by the installed DeepEP binding.
- DeepGEMM uses contiguous HIPC for HT and masked HIPC for LL, MODE=1002.
  HT/LL views share one packed W4 weight owner. Step5's post-SiLU clamp at 7
  and BF16 rounding are retained.
- The checkpoint's three BF16 expert layers are converted locally to channel
  FP8 on load. Their LL wire data stays BF16 and is quantized to E4M3FN after
  dispatch; E4M3FNUZ is never reinterpreted as E4M3FN.
- DeepEP dispatch handles are paired with their matching combine. The framework
  caches compatible communication allocations; routing metadata is recomputed
  per layer. One fenced fused-scatter cache is shared by successive HT layers.
- Shared experts retain their TP weight shards: one TP ReduceScatter sums the
  shared partials onto token owners; the layer's final TP AllGather restores
  token order. DeepEP routed results are not TP/EP-reduced a second time.
  With TP=1 these two TP collectives are absent.

## Validation

On gfx938 with DeepEP `1.1.1+dtk2610.torch2110.2610301601.g54d9ab` and
DeepGEMM `2.1.0+das.dtk2610.4e0b87b.gfx938.co5`:

- CPU/runtime regression: 180 tests, plus 20 focused configuration and
  communicator checks after adding TP-only auto support.
- Eight-rank native INT8 HT/LL and CUDA Graph checks passed. Fused scatter with
  44 experts per rank matched ordinary HT exactly, including changed routes.
- Real Step5 W4A8 and mixed FP8 blocks passed HT/LL output, LL CUDA Graph and
  shared-expert reference checks at TP/DP = 1/8, 2/4, 4/2, 8/1.
- TP2/DP2/PP2 block/group checks passed with 88 local experts per EP4 rank.
  This checks grouped layer execution, not a full pipeline server run.

The standalone block probe uses synthetic identity projections through the
real checkpoint loaders. These checks do not establish full-model dataset
accuracy, end-to-end throughput or comparative ITL against another framework.

Run the eight-rank block and communication audit in the prepared container:

```bash
VLLM_HCU_STEP5_W4A8_PROFILE=auto VLLM_HCU_STEP5_FP8_BF16_MOE=1 \
VLLM_HCU_STEP5_W4A8_GEMM_BACKEND=deepgemm VLLM_HCU_USE_CUSTOM_OPS=1 \
HCU_STEP5_DEEPEP_FUSED_SCATTER=1 ROCSHMEM_IPC_MNVL=1 \
ROCSHMEM_IPC_MNVL_NPES=16 DEEPEP_ENABLE_LL_LAYERED_OPT=1 \
ACCUMULATE_DISPATCH_PROFILE_LEVEL=0 OMP_NUM_THREADS=1 \
STEP5_TEST_TP=4 STEP5_TEST_PP=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmarks/step5_w4a8_pd/validate_deepep_block.py /tmp/step5-deepep-block-check
```

The fixture checks one native dispatch/combine pair per layer and phase,
sender token rows equal to `ceil(DP_batch_rows/TP)`, and no DP collectives,
extra TP/EP AllReduce or naive router-logit AllGather. It also compares routed
and shared outputs with an independent quantized identity-weight reference.
