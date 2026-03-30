"""This generates the test audio from the ground truth text using edge-tts

This is part of the voice input comparison pipeline for Layer 2
It creates synthetic audio files from known text so that the speech to text models
(i.e., Whisper, Voxtral) can be tested against a known ground truth using WER

the flow is as follows:
1. for every test query (defined in "config.py") generate an MP3 with edge-tts
2. then convert MP3 to WAV (16kHz, mono, 16-bit PCM) using ffmpeg, for Whisper's expected format
3. then delete the intermediate MP3
4. and return the list of generated WAV paths

Output is: data/audio/raw/<query_id>.wav
"""

# import libraries
from __future__ import annotations
import asyncio
import subprocess
from pathlib import Path
import edge_tts
from app.voice.config import ALL_QUERIES, VoiceConfig
from app.voice.models import TestQuery


# constants

# British English neural voice, this matches the regulatory text spelling
VOICE = "en-GB-SoniaNeural"


# this is an internal async function: to generate audio for a single query


async def _generate_single(query: TestQuery, output_path: Path) -> None:
    """This generates the WAV audio for one query with edge-tts

    Steps:
    1. saves edge-tts output as a temp MP3
    2. converts MP3 to WAV using ffmpeg (16kHz, mono, s16 PCM)
    3. finally, deletes the temporary MP3

    Args:
    query: the TestQuery whose "ground_truth" text will be spoken
    output_path: the full path for the final WAV file (e.g., data/audio/raw/a1.wav)
    """
    mp3_path = output_path.with_suffix(".mp3")

    communicate = edge_tts.Communicate(query.ground_truth, VOICE)
    await communicate.save(str(mp3_path))

    # ffmpeg flags: -y overwrite, -ar 16000 sample rate, -ac 1 mono, -sample_fmt s16 PCM
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp3_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )

    mp3_path.unlink()


# public function to generate test audio for all the queries


def generate_test_audio(config: VoiceConfig) -> list[Path]:
    """this generates WAV files for all the test queries and returns a list of created paths

    it skips queries where the WAV file already exists
    the generated files feed into "preprocessing.py" -> whisper/voxtral transcribers -> "metrics.py"

    Args:
    config: VoiceConfig which provides the directory paths and "raw_audio_path()"
    Returns:
    a list of 10 Path objects that point to the generated WAV files
    """
    config.ensure_dirs()

    paths: list[Path] = []

    for query in ALL_QUERIES:
        wav_path = config.raw_audio_path(query.query_id)

        if wav_path.exists():
            print(f" EXISTS {query.query_id}: {wav_path.name}")
            paths.append(wav_path)
            continue

        print(
            f" Generating {query.query_id} ({query.category}): {query.ground_truth[:60]}..."
        )

        # "asyncio.run()"" bridges sync to async to call edge-tts async function
        asyncio.run(_generate_single(query, wav_path))

        paths.append(wav_path)
        print(f"    -> {wav_path.name}")

    return paths
