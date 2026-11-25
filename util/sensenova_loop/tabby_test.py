"""
SenseNova-Loop-9B through a running tabbyAPI: chat, reasoning split, tool call, image, speed.

    (start TabbyAPI with the model loaded first)
    python util/sensenova_loop/tabby_test.py [--url http://127.0.0.1:5000] [--long 40000]
"""
import argparse, base64, io, json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LONG_TEXT = os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt")


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout = 1200) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def show(name, out, dt):
    m = out["choices"][0]["message"]
    u = out.get("usage", {})
    speed = ", ".join(f"{k} {u[k]:.1f}" if isinstance(u[k], float) else f"{k} {u[k]}"
                      for k in ("prompt_tokens", "completion_tokens", "prompt_tokens_per_sec", "completion_tokens_per_sec")
                      if u.get(k) is not None)
    print(f"\n{name}: {speed}, wall {dt:.1f} s")
    if m.get("reasoning_content"):
        print(f"  reasoning: {m['reasoning_content'][:300]!r}")
    if m.get("tool_calls"):
        print(f"  tool_calls: {json.dumps([t['function'] for t in m['tool_calls']])}")
    print(f"  content: {(m.get('content') or '')[:500]!r}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default = "http://127.0.0.1:5000")
    ap.add_argument("--long", type = int, default = 40000, help = "characters / 4 = rough tokens of the long prompt")
    args = ap.parse_args()
    base = {"temperature": 0, "max_tokens": 400}
    off, on = {"chat_template_kwargs": {"enable_thinking": False}}, {"chat_template_kwargs": {"enable_thinking": True}}

    out, dt = post(args.url, base | off | {"messages": [{"role": "user", "content": "What is the capital of Australia? One sentence."}]})
    show("thinking off", out, dt)

    out, dt = post(args.url, base | on | {"max_tokens": 1200, "messages": [
        {"role": "user", "content": "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. What does the ball cost?"}]})
    show("thinking on", out, dt)

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}}]
    msgs = [{"role": "user", "content": "What is the weather in Dhaka right now, in Celsius?"}]
    out, dt = post(args.url, base | off | {"messages": msgs, "tools": tools})
    m = show("tool call", out, dt)
    if m.get("tool_calls"):
        msgs += [m, {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"],
                     "content": json.dumps({"city": "Dhaka", "temp_c": 27, "sky": "haze"})}]
        out, dt = post(args.url, base | off | {"messages": msgs, "tools": tools})
        show("after the tool result", out, dt)

    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([40, 40, 200, 200], fill = "red")
        d.ellipse([250, 250, 410, 410], fill = "blue")
        buf = io.BytesIO(); img.save(buf, "PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        out, dt = post(args.url, base | off | {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "Which shapes are in this image, what colour is each, and where is each?"}]}]})
        show("image", out, dt)
    except ImportError:
        print("\nimage: skipped (no PIL)")

    for name, q in (("code", "Write a Python class for a min-heap with push, pop and peek, with docstrings."),
                    ("prose", "Write a short story about a lighthouse keeper who finds a message in a bottle.")):
        out, dt = post(args.url, base | off | {"messages": [{"role": "user", "content": q}]})
        show(name, out, dt)

    if args.long:
        with open(LONG_TEXT, encoding = "utf8") as f:
            story = f.read()
        story = (story * (args.long * 4 // len(story) + 1))[:args.long * 4]
        cut = len(story) // 10
        needle = "\n\nNote from the archivist: the vault code is 7-4-1-9-2-6.\n\n"
        text = story[:cut] + needle + story[cut:]
        out, dt = post(args.url, base | off | {"max_tokens": 60, "messages": [
            {"role": "user", "content": text + "\n\nWhat is the vault code mentioned by the archivist? Answer with the code only."}]})
        show("long prompt (needle at 10%)", out, dt)


if __name__ == "__main__":
    main()
