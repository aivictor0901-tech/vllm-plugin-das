#!/usr/bin/env bash
# Uses the vLLM image's native vllm.benchmarks.serve and RandomDataset.
set -euo pipefail
export PYTHONPATH="/my_code/vllm-plugin-das${PYTHONPATH:+:$PYTHONPATH}"
RUN_ID=${RUN_ID:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
RESULT_DIR=${RESULT_DIR:-/my_code/pd_log/ep16dp16_prefill}
mkdir -p "$RESULT_DIR"
exec >> "$RESULT_DIR/${RUN_ID}_prefill_128k_bs32.log" 2>&1
printf 'Waiting for EP16 health: %s\n' "$(date --iso-8601=seconds)"
python -u - <<'PY'
import time, urllib.request
for attempt in range(540):
    try:
        with urllib.request.urlopen('http://127.0.0.1:18666/health', timeout=3) as response:
            if response.status == 200:
                print('Service healthy; starting prefill benchmark', flush=True)
                break
    except Exception:
        pass
    time.sleep(5)
else:
    raise SystemExit('Service did not become healthy within 45 minutes')
PY
if [[ ${RUN_SMOKE:-0} == 1 ]]; then
  python /my_code/pd_startbash/smoke_step5.py > "$RESULT_DIR/${RUN_ID}_smoke.jsonl" 2>&1
  printf 'Arithmetic smoke check passed; starting 128K benchmark.\n'
fi
vllm bench serve \
 --backend vllm --base-url http://127.0.0.1:18666 \
 --endpoint /v1/completions \
 --model step5 --tokenizer /models/Step-5-Preview-Channel-INT4-w4a8-INT8-w8a8 \
 --served-model-name step5 \
 --dataset-name random --random-input-len 131072 --random-output-len 1 \
 --random-range-ratio 0 --num-prompts 32 --max-concurrency 32 \
 --temperature 0 --num-warmups 1 --request-rate inf --ignore-eos --seed 42 \
 --extra-body '{"skip_special_tokens": false}' \
 --percentile-metrics ttft,e2el --save-result --save-detailed \
 --result-dir "$RESULT_DIR" --result-filename "${RUN_ID}_prefill_128k_bs32.json"

python - "$RESULT_DIR/${RUN_ID}_prefill_128k_bs32.json" <<'CHECK'
import json, sys
with open(sys.argv[1]) as f:
    result = json.load(f)
if result['completed'] != 32 or result['failed'] != 0:
    raise SystemExit(f"BENCHMARK FAILED: completed={result['completed']} failed={result['failed']}")
print(json.dumps({
    'completed': result['completed'],
    'input_tokens': result['total_input_tokens'],
    'input_lens': result['input_lens'],
    'prefill_tokens_per_second': result['total_input_tokens'] / result['duration'],
    'mean_ttft_s': sum(result['ttfts']) / len(result['ttfts']),
    'max_ttft_s': max(result['ttfts']),
    'duration_s': result['duration'],
}, ensure_ascii=False, indent=2))
CHECK
