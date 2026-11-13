#!/usr/bin/env python3
"""Original deployment smoke check using the OpenAI-compatible vLLM HTTP API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18666')
    parser.add_argument('--concurrency', type=int, default=16)
    args = parser.parse_args()

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            args.url + path, data=data,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            body = response.read()
            return json.loads(body) if body else {'status': response.status}

    print(json.dumps({'health': request('/health'), 'models': request('/v1/models')}, ensure_ascii=False), flush=True)

    def check(i):
        expected = str(i + 7)
        # Use this checkpoint's ChatML delimiters and close the reasoning block
        # so a short answer tests actual decode without a long reasoning budget.
        prompt = (
            '<|im_start|>user\n'
            f'计算 {i}+7，只输出结果数字，不要解释。'
            '<|im_end|>\n<|im_start|>assistant\n<think></think>'
        )
        started = time.monotonic()
        response = request('/v1/completions', {
            'model': 'step5', 'prompt': prompt, 'temperature': 0,
            'max_tokens': 32,
        })
        actual = response['choices'][0]['text'].strip()
        result = {'case': i, 'expected': expected, 'actual': actual,
                  'seconds': round(time.monotonic() - started, 3),
                  'response': response, 'passed': actual == expected}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    first = check(1)
    if not first['passed']:
        raise SystemExit('Single-request correctness check failed')
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(check, range(2, args.concurrency + 2)))
    passed = sum(result['passed'] for result in results)
    print(json.dumps({'concurrent_passed': passed, 'concurrent_total': len(results)}), flush=True)
    if passed != len(results):
        raise SystemExit('Concurrent correctness check failed')


if __name__ == '__main__':
    main()
