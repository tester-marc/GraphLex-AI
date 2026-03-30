"""this is for the Whisper transcription across the different model sizes

It provides local speech to text using OpenAI's Whisper
and it's part of the GraphLex AI voice input comparison (for Layer 2)

it tests 4 Whisper sizes: tiny (39M), base (74M), small (244M), and medium (769M params)

Result: Whisper small was chosen for production use. See report for more information
and details.

Design decisions:
1. model caching: the models load once per process
2. CPU mode: fp16=False is always set
3. Context biasing: "initial_prompt" with regulatory vocabulary
"""

# import libraries
from __future__ import annotations
import time
from pathlib import Path
import whisper

# WHISPER_CONTEXT_PROMPT: regulatory vocabulary hint passed as initial_prompt
# WHISPER_MODELS: list of the sizes to test ["tiny", "base", "small", "medium"]
from app.voice.config import WHISPER_CONTEXT_PROMPT, WHISPER_MODELS


# Model cache
# this maps the model size names to the loaded Whisper objects
_model_cache: dict[str, whisper.Whisper] = {}


def _load_model(size: str) -> whisper.Whisper:
    """ini order to load a Whisper model by size and using the cache to avoid redundant loads

    Args:
    size: either "tiny", "base", "small", or "medium"

    Returns:
    whisper.Whisper: the loaded model ready to call ".transcribe()" on
    """
    if size not in _model_cache:
        print(f" Loading whisper-{size}...")
        _model_cache[size] = whisper.load_model(size)
    return _model_cache[size]


def transcribe(
    audio_path: Path,
    model_size: str,
    context_biasing: bool = False,
) -> tuple[str, float]:
    """This transcribes an audio file to text using the specific Whisper model size

    Args:
    audio_path: the path to the WAV file, either raw or
    preprocessed (16kHz mono, ffmpeg-normalized)
    The comparison tests both in order to measure the impact that preprocessing has
    model_size: either "tiny", "base", "small", or "medium"
    context_biasing: if true, this passes WHISPER_CONTEXT_PROMPT as
    "initial_prompt", and biases the decoder towards the regulatory vocabulary
    (e.g., "FADP", "GDPR", "EDPB")

    Returns:
    a tuple of (transcription_text, latency_ms)
    """
    model = _load_model(model_size)

    kwargs: dict = {
        "language": "en",  # skips auto detection
        "fp16": False,  # CPU mode
    }

    if context_biasing:
        kwargs["initial_prompt"] = WHISPER_CONTEXT_PROMPT

    start = time.perf_counter()
    result = model.transcribe(
        str(audio_path), **kwargs
    )  # Whisper expects str, not Path
    latency_ms = (time.perf_counter() - start) * 1000

    # "".strip()"" removes the whitespace that Whisper sometimes adds
    return result["text"].strip(), latency_ms


def available_models() -> list[str]:
    """This returns the Whisper model sizes which are available to be compared

    Returns:
    list[str]: e.g., ["tiny", "base", "small", "medium"]
    """
    return list(WHISPER_MODELS)
