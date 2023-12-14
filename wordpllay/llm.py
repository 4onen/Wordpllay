"""
This module provides a wrapper around our calls out to the language model.
"""
import logging
from pathlib import Path
import asyncio
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
)
from llama_cpp import Llama


class PromptFormat(NamedTuple):
    """
    A prompt format is a pair of a name and a format string. The format string
    should contain a single {instruction} placeholder, which will be replaced
    with the instruction.

    We do not accept more complex formats because I'm lazy and I've seen only
    limited utility for them for plain assistant models.
    """

    name: str
    format: str

    def with_instruction(self, instruction: str) -> str:
        """
        Returns a string with the instruction formatted into the format string.
        """
        return self.format.format(instruction=instruction)

    def __str__(self) -> str:
        return self.name


ALPACA_PROMPT_FORMAT = PromptFormat(
    "Alpaca",
    """### Instruction:
{instruction}

### Response:
""",
)

KNOWN_PROMPT_FORMATS: Dict[str, PromptFormat] = {
    "chupacabra": ALPACA_PROMPT_FORMAT,
}


def prompt_format_guess(model_path: Path) -> PromptFormat:
    """
    Guesses the prompt format for a model identifier. This is a heuristic and
    may not be correct. If no format can be guessed, a default, common format
    is returned.
    """
    model_identifier = model_path.name

    prompt_format = None

    for model_identifier_fragment, new_format in KNOWN_PROMPT_FORMATS.items():
        if model_identifier_fragment in model_identifier:
            if prompt_format is not None:
                logging.warning(
                    "Multiple prompt formats match model identifier! Switching from %s to %s",
                    prompt_format,
                    new_format,
                )
            prompt_format = new_format

    if prompt_format is None:
        prompt_format = ALPACA_PROMPT_FORMAT
        logging.warning("No prompt format guess! Assuming %s.", prompt_format)
    else:
        logging.info("Prompt format guess: %s", prompt_format)

    return prompt_format


class LLM:
    """
    A wrapper around the language model.
    """

    __slots__ = ("model", "prompt_format")

    def __init__(self, model_path: Path):
        self.prompt_format = prompt_format_guess(model_path)
        self.model = Llama(model_path=str(model_path))

    async def generate_async(
        self,
        instruction: str,
        chunk_writer: Optional[Callable[[str], Any]] = None,
        **kwargs,
    ) -> str:
        """
        Generates a response asynchronously. If chunk_writer is provided, it
        will be called with each chunk of the response as it is generated.
        """
        prompt = self.prompt_format.with_instruction(instruction)
        if chunk_writer is not None:
            assert asyncio.iscoroutinefunction(chunk_writer)
            accumulated_text = ""
            for result in self.model(prompt, stream=True, **kwargs):
                print(result)
                text = result["choices"][0]["text"]
                accumulated_text += text
                await chunk_writer(text)
                if asyncio.current_task().cancelled():
                    break
            return accumulated_text

        result = self.model(prompt, **kwargs)
        return result["choices"][0]["text"]


async def print_async(text: str) -> None:
    """
    Prints text asynchronously.
    """
    print(text)


if __name__ == "__main__":
    llm = LLM(
        Path(
            "/home/mdupree/oobabooga_linux/models/chupacabra-7b-v3.Q4_K_M.gguf"
        )
    )

    print(
        asyncio.run(
            llm.generate_async(
                '### Instruction:\nSay "Moo" and nothing else.\n\n### Response:',
                max_tokens=16,
                chunk_writer=print_async,
            )
        )
    )

    print("Done.")
