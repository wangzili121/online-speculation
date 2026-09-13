"""Local tokenizer JSON only; no model code, remote code, or implicit chat template."""

from pathlib import Path
import json


class LocalTokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer
        self.directory = Path(directory)
        self.backend = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        self._chat_tokenizer = None

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def render_chat(self, messages, *, add_generation_prompt=False):
        """Render the model's local template using a generic tokenizer utility.

        No AutoModel, auto-mapped tokenizer class or remote model code is loaded.
        The HF dependency here is template handling only, not model execution.
        """
        if self._chat_tokenizer is None:
            from transformers import PreTrainedTokenizerFast
            raw = json.loads((self.directory / "tokenizer_config.json").read_text(encoding="utf-8"))
            template_path = self.directory / "chat_template.jinja"
            template = template_path.read_text(encoding="utf-8") if template_path.exists() else raw.get("chat_template")
            if not isinstance(template, str) or not template:
                raise ValueError("an explicit local chat template is required")
            special = {}
            for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
                value = raw.get(key)
                if isinstance(value, dict):
                    value = value.get("content")
                if isinstance(value, str):
                    special[key] = value
            self._chat_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(self.directory / "tokenizer.json"), chat_template=template, **special)
        return self._chat_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
