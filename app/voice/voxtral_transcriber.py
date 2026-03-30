"""for the Voxtral transcription via the Mistral API

This provides the speech to text transcription using Mistral's Voxtral Mini model (4.7B params)

In GraphLex AI, this module is part of the voice input comparison (for Layer 2) which compares:
- OpenAI Whisper (local and tested on tiny/base/small/medium)
- Mistral Voxtral Mini (via the cloud API)

Voxtral Mini was tested and ultimately rejected:
This rejection is documented in the final report in detail.

Exports:
- transcribe()     : this takes an audio file, and returns text and latency
- is_available()   : checks whether the API key is configured
- VOXTRAL_MODEL    : the model identifier string used in the API calls
"""

# import libraries
from __future__ import annotations  # for type hint syntax
import os
import time
from pathlib import Path
from mistralai.client import Mistral

# the regulatory terms (e.g., "FADP", "GDPR", "EDPB") passed as context bias hints to
# the API
from app.voice.config import VOXTRAL_CONTEXT_BIAS


VOXTRAL_MODEL = "voxtral-mini-latest"


def _get_client() -> Mistral:
    """this creates and returns an auth Mistral API client

    Raises:
    RuntimeError: if the MISTRAL_API_KEY isn't set in the env
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    return Mistral(api_key=api_key)


def transcribe(
    audio_path: Path,
    context_biasing: bool = False,
) -> tuple[str, float]:
    """in order to transcribe an audio file to text using Mistral's Voxtral Mini API

    it is called by the comparison harness ("app/voice/comparison.py") for every test query in
    up to 4 configurations as follows:
    with / without ffmpeg preprocessing x with / without context biasing
    The results are evaluated using WER, Entity WER, and the Fabricated Insertion Count

    Args:
    audio_path: this is the path to a raw or pre-processed WAV file
    context_biasing: if true, it passes VOXTRAL_CONTEXT_BIAS
    to the API in order to improve recognition of domain specific terms

    Returns:
    a tuple of:
    - transcription_text (str): the stripped transcribed text
    - latency_ms (float): the round trip time in ms
    """
    client = _get_client()
    audio_data = audio_path.read_bytes()

    kwargs: dict = {"language": "en"}
    if context_biasing:
        kwargs["context_bias"] = VOXTRAL_CONTEXT_BIAS

    start = time.perf_counter()

    result = client.audio.transcriptions.complete(
        model=VOXTRAL_MODEL,
        file={
            "file_name": audio_path.name,  # this is used by the API for detection of the format
            "content": audio_data,
        },
        **kwargs,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return result.text.strip(), latency_ms


def is_available() -> bool:
    """this return true if the MISTRAL_API_KEY is configured, and otherwise false

    it is used by the comparison harness to skip Voxtral if no API key is there,
    since Whisper runs locally but Voxtral requires a paid Mistral API key
    """
    return bool(os.environ.get("MISTRAL_API_KEY"))
