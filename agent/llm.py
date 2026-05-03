"""Qwen3.5-2B Q4_K_M wrapper. Builds the Qwen ChatML prompt and runs llama-cpp."""

from pathlib import Path
from typing import Iterator

from llama_cpp import Llama, LlamaGrammar

QWEN_STOP = ["<|im_end|>"]


def build_prompt(system: str, history: list[dict]) -> str:
    out = f"<|im_start|>system\n{system}<|im_end|>\n"
    for msg in history:
        out += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out


class Agent:
    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 4):
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )

    def generate(
        self,
        system: str,
        history: list[dict],
        max_tokens: int = 256,
        grammar: LlamaGrammar | None = None,
        temperature: float = 0.2,
    ) -> str:
        prompt = build_prompt(system, history)
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=QWEN_STOP,
            grammar=grammar,
            temperature=temperature,
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    def stream(
        self,
        system: str,
        history: list[dict],
        max_tokens: int = 768,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        prompt = build_prompt(system, history)
        for chunk in self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=QWEN_STOP,
            temperature=temperature,
            echo=False,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token
