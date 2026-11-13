"""QwenTokenizer wrapper in pure Python/transformers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer


class QwenTokenizer:
    def __init__(self, token_path: str | Path, skip_special_tokens: bool = True):
        special_tokens = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "additional_special_tokens": [
                "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
                "[breath]", "<strong>", "</strong>", "[noise]",
                "[laughter]", "[cough]", "[clucking]", "[accent]",
                "[quick_breath]", "[chuckle]", "[clearing_throat]",
                "[cry]", "[gasp]", "[heavy_breath]", "[scream]",
                "[sneer]", "[sob]",
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]", "[sneeze]", "[inhale]", "[sniffle]", "[exhale]", "[giggle]", "[yawn]",
                "<sys_begin>", "<sys_end>", "<prompt_begin>", "<prompt_end>", "<|BOU|>", "<|BOA|>", "<eos>",
                "[TEXT]", "[SPEECH]", "[beginofspeech]", "[endofspeech]",
            ],
        }
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(token_path))
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer([text], return_tensors="np")
        return tokens["input_ids"][0].tolist()

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)
