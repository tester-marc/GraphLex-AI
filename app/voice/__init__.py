# Voice Input Layer for the Whisper and Voxtral comparison
#
# this is "__init__.py" for the "app.voice" package (which is Layer 2 of GraphLex AI)
# it marks the dir as a Python package and defines the public API
# this way callers can write:
#
# "from app.voice import VoiceConfig, TranscriptionResult"
#
# instead of importing from the internal modules
#
# the package contains:
# __init__.py               : this file
# __main__.py               : the CLI entry point "python -m app.voice"
# models.py                 : the data classes: TestQuery, TranscriptionResult,
#                             VoiceComparisonResult
# config.py                 : the test queries, model lists, vocabulary, and paths
# audio_generator.py        : this generates synthetic test audio with edge-tts
# preprocessing.py          : ffmpeg audio normalization pipeline
# whisper_transcriber.py    : OpenAI Whisper transcription
# voxtral_transcriber.py    : Mistral Voxtral Mini transcription (via cloud API)
# metrics.py                : WER, Entity WER, FIC calculation
# comparison.py             : the comparison harness that orchestrates all
#
# this comparison tested Whisper (models: tiny/base/small/medium) vs. Voxtral Mini
# for 10 queries, 2 pre-processing modes, and 2 biasing modes
# Ultimately, Whisper small was chosen for production use (for details on why please
# see final report)

# TranscriptionResult:
# the output of a single transcription attempt
# fields:
# model_name, query_id, category ("A"/"B"), ground_truth, transcription,
# preprocessing ("raw"/"ffmpeg"), context_biasing (bool), latency_ms,
# general_wer, entity_wer, fabricated_insertion_count
#
# VoiceComparisonResult:
# the aggregated results for one model config,
# which are averaged across all 10 test queries and split down by category (A vs. B)
from app.voice.models import TranscriptionResult, VoiceComparisonResult

# VoiceConfig:
# this manages all the file paths (project_root, audio_dir, raw_dir,
# preprocessed_dir, output_dir) and provides these helpers:
# ensure_dirs(), raw_audio_path(query_id),
# preprocessed_audio_path(query_id), results_path()
from app.voice.config import VoiceConfig

# "__all__" defines the public interface of this package
__all__ = [
    "VoiceConfig",
    "TranscriptionResult",
    "VoiceComparisonResult",
]
