"""Chat requests against a running tabbyAPI: tokens per second of the answer (from the usage block), greedy."""
import json, sys, time, urllib.request
PROMPTS = [
    ("math", "What is 84 * 3 / 2? Answer briefly."),
    ("code", "Write a Python function that returns the n-th Fibonacci number, with a docstring and type hints."),
    ("prose", "Explain in about 120 words why the sky is blue."),
    ("list", "List the planets of the solar system in order from the sun, one per line, with one fact each."),
    ("essay", "Write a 250-word essay about the history of the printing press."),
    ("code2", "Write a Python class for an LRU cache with get and put, using only the standard library. Include a short usage example."),
]
url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000") + "/v1/chat/completions"
for rnd in range(2):
    for name, p in PROMPTS:
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout = 600)); dt = time.time() - t0
        u = r["usage"]; text = r["choices"][0]["message"].get("content") or ""
        if rnd:
            print(f"{name:6s}: {u['completion_tokens']:4d} tokens, {u.get('completion_tokens_per_sec', u['completion_tokens'] / dt):6.1f} T/s, wall {dt:5.1f} s | {text[:70]!r}", flush = True)
