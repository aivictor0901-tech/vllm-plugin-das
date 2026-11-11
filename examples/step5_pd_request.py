"""Diagnostic P->D request using the existing Mooncake request fields.

Both services must enable step5_pd with identical model/tokenizer, TP and MTP
settings. This diagnostic client fixes the selected DP ranks explicitly.
"""

import argparse
import json
import urllib.request
import uuid


def request(url, payload=None, rank=None):
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    req = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(), headers=headers,
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def run(args):
    messages = [{"role": "user", "content": args.prompt}]
    tokens = request(args.prefill.rstrip("/") + "/tokenize", {
        "model": args.model, "messages": messages, "add_generation_prompt": True,
    })["tokens"]
    bootstrap = args.bootstrap.rstrip("/")
    engines = request(bootstrap + "/query")
    engine = engines[str(args.p_rank)]["engine_id"]
    common = {"transfer_id": str(uuid.uuid4())}
    request(args.prefill.rstrip("/") + "/v1/completions", {
        "model": args.model, "prompt": tokens, "max_tokens": 1,
        "temperature": 0, "ignore_eos": True,
        "kv_transfer_params": {**common, "do_remote_decode": True},
    }, args.p_rank)
    return request(args.decode.rstrip("/") + "/v1/completions", {
        "model": args.model, "prompt": tokens, "max_tokens": args.max_tokens,
        "temperature": 0,
        "kv_transfer_params": {**common, "do_remote_prefill": True,
                               "remote_engine_id": engine, "remote_bootstrap_addr": bootstrap},
    }, args.d_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True, help="P API URL")
    parser.add_argument("--bootstrap", required=True, help="P Mooncake bootstrap URL")
    parser.add_argument("--decode", required=True, help="D API URL")
    parser.add_argument("--model", default="step5")
    parser.add_argument("--prompt", default="请简短回答：1+1等于多少？")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--p-rank", type=int, default=0)
    parser.add_argument("--d-rank", type=int, default=0)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
