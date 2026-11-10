# Step5 GPTQ W4A8：Prefill 与 Decode

容器：`lijing_step5_w4a8_deepgemm_1110`，镜像 ID `17e48f1f88e8`。
模型：`/model/step-5-preview-channel-w4a8-GPTQ`，从 S5 本地 NVMe 的同名路径只读挂载。
保留模型原始 `quant_method=compressed-tensors`，不修改权重文件。

```bash
docker exec -it lijing_step5_w4a8_deepgemm_1110 bash
bash /workspace/step5-w4a8/vllm-plugin/benchmarks/step5_w4a8_pd/serve_gptq_decode.sh --dry-run
# 确认有八张可用卡后，去掉 --dry-run 启动服务。
```

入口固定 TP4、DP2、EP8、PP1，使用 allgather_reducescatter 和 DeepGEMM W4A8 HIPC。
默认端口 18134，长度上限 139264，chunk 8192，显存预算 0.85。
可通过 `PORT`、`MAX_MODEL_LEN`、`CHUNK_SIZE`、`MAX_NUM_SEQS`、`GPU_MEMORY_UTILIZATION` 覆盖对应配置。
默认使用真实 Prefill 和真实 MTP3 验证，不使用 fake-prefill 或 synthetic 接受率。
性能实验需要 synthetic 接受率时，显式通过 `SPECULATIVE_CONFIG` 传入完整 JSON。

关键环境变量：

```bash
export VLLM_HCU_STEP5_W4A8_PROFILE=decode
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_W4A8_GEMM_BACKEND=deepgemm
```

第二个变量为 `1` 时：85 层专家保留 INT4 权重、动态 INT8 激活；原第 88、89、90 层 BF16 专家在加载时按输出通道量化为 FP8，激活动态量化为 FP8，scale 为 FP32。
设为 `0` 时，这三层保留原有 BF16 专家路径。
只有本入口明确选择的 Step5 GPTQ 配置会启用适配，原 FP8 和 SlimQuant 路径保持各自的选择逻辑。

GPTQ 的 `weight_packed` 在加载时转换为已有算子需要的 signed INT4 high-nibble-first 字节，并将 scale 除以 16，补偿 W4A8 算子内部的 INT8 域表示。
转换在本 rank 实际拥有的专家上进行，随后复用现有原位 HIPC 打包；原始 `weight_shape` 做维度校验后不作为运行参数保留。
每层检查本 rank 的全部专家权重与 scale 已加载，模型检查 INT4 packed storage 的唯一所有权，以及 FP8/BF16 层的实际 dtype。

Decode 支持固定 TP4 / DP2 / EP8，Prefill 支持固定 TP1 / DP8 / EP8，均为 PP1。
不启用 EPLB、动态权重重载、DeepEP 或 LoRA。
此适配不等于完成了全模型精度评测；将 BF16 专家转 FP8 会产生额外量化误差。

## step-w4a8-nmz Decode 适配与验证

基于 `step-w4a8-nmz` 的 `efae5be5982631b46e37c3ae33cfd691eadd9ea0` 接入，保留该分支已有的通用 compressed-tensors W4A8、权重缓存刷新和图执行修复。
Step5 的 SwiGLUStep7 与 85 层 INT4 / 3 层 FP8 混合加载由专用适配层处理；其他模型继续使用原有量化选择逻辑。
本部署使用原 AG/RS 通信：INT8 激活 AllGather、DeepGEMM contiguous HIPC G1/G2（MODE=1002）、原专家 combine/reduce；不包含实验性的 Step5 DeepEP LL 接入。

验证环境为 S5 单机八卡 BW1101（BW1100 平台）、容器 `lijing_step5_w4a8_deepgemm_1110`。
torch、vLLM、DeepGEMM 和 LightOp 依赖保留原版本，只重新编译和安装 plugin。
plugin 构建保留 `gfx938`、Code Object V5 和可选的 Dense FP8 原生扩展；W4A8 G1/G2 使用已验证的外部 `4e0b87b.gfx938.co5` DeepGEMM wheel。

158 项针对性测试通过，覆盖 GPTQ 编码与尺度、混合精度选择、DeepGEMM 自定义算子、部署隔离，以及新分支通用 CT W4A8 / masked ABI 兼容性。
真实 Prefill、真实 MTP3、FULL_AND_PIECEWISE CUDA Graph 的 GSM8K 前 64 题回归无请求错误、无输出截断。
沿用冻结判分器为 62/64；其中第 7 题同时给出 `160` 分钟和 `2 小时 40 分钟` 两个等价 boxed 答案，最后一个格式未被数字判分器解析。
保留原始判分不变，另做等价单位核对后为 63/64，与旧版本的 63/64 一致；这是小样本回归，不是完整 GSM8K 分数。

Decode 性能使用 fake-prefill、131072 输入 token、4096 输出 token、MTP3 synthetic 接受率 `[0.838992,0.598322,0.362686]`。
TP4 / DP2 / EP8，chunk=1024，长度上限 262144，max-num-seqs=64，显存预算 0.90；图模式与前述部署相同，capture size=256，compile sizes 为 16/64。
每组并发一轮完整预热、一轮正式测试；若正式轮出现 JIT，则保留并剔除该轮，复用预热后重测。
吞吐为所有请求共同活跃的 Decode 窗口内生成 token 数除以窗口时长；ITL 为该窗口内流式 chunk 的平均间隔，MTP 每个 chunk 可包含多个 token。
正式计时窗口须无抢占、无 JIT、无 fake KV 填充，也不开 profiler。

2026-11-11 实测结果如下，旧版为保留的先前测量；各点正式轮均无抢占、无 JIT：

| 总并发 | 旧 W4A8 AG/RS 吞吐 (tok/s) | 本分支吞吐 (tok/s) | 旧 ITL (ms) | 本分支 ITL (ms) |
| --- | --- | --- | --- | --- |
| 32 | 1504.59 | 1502.15 | 59.86 | 59.95 |
| 64 | 2267.58 | 2267.26 | 79.04 | 79.03 |

## GPTQ Prefill：TP1 / DP8 / EP8

Prefill 入口使用同一 GPTQ 模型与 compressed-tensors 加载适配器，显式选择
`VLLM_HCU_STEP5_W4A8_PROFILE=prefill`，并校验固定 TP1 / DP8 / EP8、PP1。
W4A8 G1/G2 沿用 DeepGEMM contiguous HIPC MODE=1002 和原 AG/RS 通信，
85 层 W4A8 与 3 层 BF16-to-FP8 的权重选择逻辑与 Decode 一致。

在安装好 plugin、DeepGEMM 及模型的容器中执行：

```bash
bash /workspace/step5-w4a8/vllm-plugin/benchmarks/step5_w4a8_pd/serve_gptq_prefill.sh --dry-run
# 检查输出后，去掉 --dry-run 启动。
```

默认端口 18132，chunk=8192，max-model-len=262144，max-num-seqs=64，
显存预算 0.875，FP8 E4M3 KV cache，关闭 prefix caching。
图模式为 FULL_AND_PIECEWISE，capture cap=64，compile sizes=16/64；
默认真实 Prefill、真实 MTP3。`SPECULATIVE_CONFIG` 可显式覆盖性能实验配置。
`CHUNK_SIZE`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`GPU_MEMORY_UTILIZATION` 和 `PORT`
可覆盖对应默认值。

入口复用 FP8 Prefill 的投影融合、FP8 Indexer、Router/Shared、DBO 和 DP MoE pipeline；
DBO、DP MoE pipeline 和 Direct Sum 的默认 token 阈值分别为 8192、16384、16384。
默认静态专家放置与 FP8 MoE 表位于入口同目录的 `assets/` 下，
也可通过 `STEP5_EXPERT_PLACEMENT`、`STEP5_FP8_MOE_TABLE` 指定外部资产。

已有 S5 Prefill 验证使用独立容器 `lijing_step5_prefill_compare_1111`，
基线为 `2b621b8a0df814a6f987560c286f636574792bcb` 加本次纳入的两处 Prefill 选择适配。
真实 131072-token 长输入回归为 8/8；16 并发、chunk=8192 的真实 Prefill trace
覆盖每卡 5 个 chunk，每卡 1700 次 W4A8 G1/G2 调用。
四套精度评测停止时已保存 1412/2431 题，属于部分结果，不作为全量通过记录。
这些是已有运行记录；本次合并后的最终源码另做 CPU 回归与 Shell 检查，未重新执行 GPU 精度或性能测试。

## DeepGEMM G1 / G2

先在服务容器中安装已验证的 CO5 wheel；不替换 torch/vLLM 依赖：

```bash
python3 -m pip install --no-deps --force-reinstall \
  /workspace/step5-w4a8/wheels/deepgemm-2.1.0+das.dtk2610.4e0b87b.gfx938.co5-cp312-cp312-linux_x86_64.whl
```

本入口默认选择 `deepgemm`。G1、G2 均调用
`deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc(..., config={"MODE": 1002})`。
IFB 的真实 Prefill 和 Decode 共用这条按专家排序、64 行对齐的路径；不切换成默认 1006 所需的 128 行布局。
权重沿用加载时唯一的 HIPC packed owner，scale 不再乘除 16。
3 层 FP8 专家、SwiGLUStep 的 clamp=7、router 权重和 combine 协议继续由已有路径处理。

调用封装为带输出 mutation 声明和 Fake 实现的
`torch.ops.vllm.step5_deepgemm_w4a8_gemm`，供 torch.compile 和 CUDA Graph 使用。
模型构建时检查 CO5/gfx938 标记，避免读完大 checkpoint 后才发现旧包不能启动 kernel。
运行日志显示 `gemm_backend=deepgemm, MODE=1002`；后端错误直接上报。

Step5 W4A8 只支持 `deepgemm`，未设置后端变量时也默认选择它。
内置的 native W4A8 GEMM、算子注册和构建开关已移除；设置 `native` 或其他后端会明确报错。
worker 和 P/D 入口在加载模型前检查 DeepGEMM CO5 依赖，不再要求内置 W4A8 算子。

S5 冷启动时 NFS 自动并发预读曾阻塞加载；可显式使用 eager：

```bash
bash /workspace/step5-w4a8/vllm-plugin/benchmarks/step5_w4a8_pd/serve_gptq_decode.sh \
  --safetensors-load-strategy eager
```

重新安装时，在容器内执行下面的构建入口，保留可选 tuned Dense FP8 原生算子，并且不替换基础 torch/vLLM 依赖。W4A8 GEMM 由上面安装的 DeepGEMM wheel 提供，无需在 plugin 中重复编译：

```bash
bash /workspace/step5-w4a8/vllm-plugin/benchmarks/step5_w4a8_pd/build_gptq_plugin.sh
```

## PR #297 优化复用

源分支已包含 `bf504f1fa09e0be14bd1a6f4c80698b290d6501d`（PR #297），无需重复合并。

| 优化 | 本入口的选择 |
| --- | --- |
| Dense FP8、SWA 融合 QK/Gate、QKV + Indexer 投影 | 启用；不依赖 routed experts 的 INT4/FP8 区别 |
| FP8 Indexer、寄存器 Top-K | 使用已有 DSA 优化实现 |
| Router / Shared Expert 融合与激活复用 | 启用；保留已有 stream 依赖和融合 combine 协议 |
| IPC AllGather | 默认启用，可搬运 INT8 或 FP8 激活及其 scale |
| Packed AllGather | `STEP5_DP_DISPATCH=packed` 选择；沿用原 M64 准入，其余形状回退普通通信 |
| Compact Host Block Tables | 启用，底层完整页表保留 |
| 有界上下文 CUDA Graph | 17152 个 region，超过该图范围按已有 DP 协议回退 eager，模型长度上限不变 |
| MTP Prefill tail | 1536 token；只优化符合条件的草稿 Prefill，不截短主模型输入 |
| Shared Expert 延后启动 | 大批次阈值 4096，复用原有合法路径 |
| 有界 EP workspace | 开启已有的 FP8 准入；INT4 继续使用自身 workspace 规划 |
| 静态专家放置 | 新增逐专家 GPTQ 支持；重命名权重、scale、shape，同时重排 Router 权重和 bias |
| Dense 外部调优 | 原生算子随容器插件构建；默认普通 LightOp，提供匹配调优表后才启用 tuned 路径 |
| DBO、DP MoE pipeline | 保留实现，Decode 入口默认关闭；大 Prefill 实验可显式覆盖阈值 |

例如选择 Packed 通信或外部专家布局：

```bash
STEP5_DP_DISPATCH=packed bash /workspace/step5-w4a8/vllm-plugin/benchmarks/step5_w4a8_pd/serve_gptq_decode.sh
STEP5_EXPERT_PLACEMENT=/path/to/placement.pt bash /workspace/step5-w4a8/vllm-plugin/benchmarks/step5_w4a8_pd/serve_gptq_decode.sh
```

静态放置文件须包含覆盖 88 个 MoE 层、352 个专家的 `physical_to_logical_map`；文件属于外部资产，未写入模型目录。
Dense 调优需要 `HCU_STEP5_TUNING_CONFIG` 指向经该 hipBLASLt 版本验证的 shape/算法表，再设置 `HCU_STEP5_DENSE_TUNING=1`。没有调优表时不宣称 tuned 算子生效。
PR 的 DP Fabric 限定 TP4/DP4；Prefill ASM G1 和专用 Triton G2 限定 E22/E23。
当前 TP4/DP2/EP8 为 E44，因此这些特定路径保持关闭或按原有条件回退。
本入口使用 AG/RS，不启用 DeepEP MNVL 专用优化。
