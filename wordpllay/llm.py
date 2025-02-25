"""
This module provides a wrapper around our calls out to the language model.
"""

import logging
import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLM:
    """
    A wrapper around the language model.
    """

    __slots__ = ("model", "prompt_format")

    def __init__(self, model: "BaseChatModel"):
        self.model = model

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
        prompt = [("user", instruction)]
        if chunk_writer is not None:
            assert asyncio.iscoroutinefunction(chunk_writer)
            accumulated_text = None
            async for result in self.model.astream(prompt, **kwargs):
                logging.debug(result)
                if result.content:
                    text: str = str(result.content)
                    if accumulated_text is None:
                        accumulated_text = result
                    else:
                        accumulated_text += result
                    await chunk_writer(text)
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    break
            return str(accumulated_text.content) if accumulated_text is not None else ""

        result = await self.model.ainvoke(prompt, **kwargs)
        return str(result.content)


async def print_async(text: str) -> None:
    """
    Prints text asynchronously.
    """
    print(text, end="")


if __name__ == "__main__":
    import langchain_openai
    import pydantic

    logging.basicConfig(level=logging.WARNING)

    llm = LLM(
        langchain_openai.ChatOpenAI(
            base_url="http://localhost:8080", api_key=pydantic.SecretStr("boo")
        )
    )

    result = asyncio.run(
        llm.generate_async(
            'Say "Moo" as many times as you like, then comment on the experience.',
            chunk_writer=print_async,
            max_tokens=16,
        )
    )
    print("\n===FINAL OUTPUT===")
    print(result)
