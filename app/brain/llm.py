"""
Local LLM Wrapper.

Uses llama.cpp with GGUF model.
"""

from __future__ import annotations

import threading
from pathlib import Path

from llama_cpp import Llama



class LocalLLM:
    """
    Local Gemma LLM using llama.cpp.
    """


    def __init__(self) -> None:

        self.lock = threading.Lock()


        self.model_path = (
            Path(__file__)
            .parent
            .parent
            .parent
            / "models"
            / "llm"
            / "gemma-3-1b-it-Q4_K_M.gguf"
        )


        if not self.model_path.exists():

            raise FileNotFoundError(
                f"LLM model not found: {self.model_path}"
            )


        self.model = Llama(

            model_path=str(
                self.model_path
            ),

            # RTX 4050 GPU acceleration
            n_gpu_layers=-1,

            # Context
            n_ctx=1024,

            # Batch
            n_batch=128,

            # CPU threads
            n_threads=8,

            verbose=False,

        )


    # -------------------------------------------------

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate complete response.
        """


        with self.lock:

            output = self.model(

                prompt,

                max_tokens=128,

                temperature=0.2,

                top_p=0.8,

                repeat_penalty=1.1,

            )


        return (

            output["choices"][0]["text"]

            .strip()

        )


    # -------------------------------------------------

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 128,
    ):
        """
        Stream response token by token.
        """


        with self.lock:


            stream = self.model(

                prompt,

                max_tokens=max_tokens,

                temperature=0.2,

                top_p=0.8,

                repeat_penalty=1.1,

                stream=True,

            )


            for chunk in stream:


                token = (

                    chunk["choices"][0]
                    ["text"]

                )


                if token:

                    yield token