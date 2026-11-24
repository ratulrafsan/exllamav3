"""
ExLlamaV3's eval/qbench_prompts.py (a fixed public set of conversations, sampled with the model's own output: the
"in-domain trace" that qbench scores quantizations on) for Inkling-2 in this environment. The script renders chat
templates through transformers; where that is not installed (TabbyAPI's environment), this renders the template
with TabbyAPI's own template code (the one the server uses) and tokenizes with ExLlamaV3. The two renderings were
compared on a plain and a tool conversation: identical. Arguments are those of qbench_prompts.py.

    TABBYAPI_DIR=<TabbyAPI checkout> python util/inkling2/qbench_trace.py -m <model dir> -o trace.json \
        --min_tokens 45000 --max_new_tokens 2560
"""
import asyncio, os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.environ["TABBYAPI_DIR"])

from common.templating import PromptTemplate
from exllamav3 import Tokenizer

_templates = {}


def hf_chat_template(self, messages, add_generation_prompt = True, embeddings = None, **template_kwargs):
    directory = self.config.directory
    tmpl = _templates.get(directory)
    if tmpl is None:
        with open(os.path.join(directory, "chat_template.jinja"), encoding = "utf8") as f:
            tmpl = _templates[directory] = PromptTemplate("chat_template", f.read())
    text = asyncio.run(tmpl.render(dict(messages = messages, add_generation_prompt = add_generation_prompt, **template_kwargs)))
    return self.encode(text, encode_special_tokens = True, add_bos = False)


Tokenizer.hf_chat_template = hf_chat_template

if __name__ == "__main__":
    script = os.path.join(REPO, "eval", "qbench_prompts.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name = "__main__")
