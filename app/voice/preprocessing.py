"""the FFmpeg audio preprocessing pipeline for the voice input

this prepares raw audio files for speech to text transcription

models like Whisper and Voxtral work best on mono 16kHz audio with consistent volume
and minimal background noise

both models are tested on both raw and preprocessed audio in order
to measure whether preprocessing actually improves the WER (Word Error Rate)

This pipeline applies 3 transformations:
1. volume normalisation: consistent loudness across all audio clips
2. format conversion: 16kHz mono WAV (which is the native Whisper format)
3. noise filtering: this removes low frequency sounds below 80Hz

the file flow is as follows:
data/audio/raw/a1.wav -> data/audio/preprocessed/a1.wav
"""

# import libraries
from __future__ import annotations
import subprocess
from pathlib import Path
from app.voice.config import ALL_QUERIES, VoiceConfig


def preprocess_audio(input_path: Path, output_path: Path) -> None:
    """apply the ffmpeg preprocessing: volume normalisation, 16kHz mono, noise filter

    Parameters:
    input_path : Path
    the path to the raw audio file
    output_path : Path
    the path for the preprocessed output

    Returns:
    None
    it raises "subprocess.CalledProcessError" if ffmpeg fails
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",  # overwrite the output without asking
            "-i",
            str(input_path),
            "-af",
            # loudnorm: EBU R128 normalization, I=-16 LUFS target, TP=-1.5 true peak, LRA=11
            # highpass=f=80: this removes noise below 80Hz that doesn't carry any speech
            "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=80",
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # mono
            "-sample_fmt",
            "s16",  # signed 16bit PCM
            str(output_path),
        ],
        capture_output=True,  # suppresses verbose ffmpeg output
        check=True,  # raises "CalledProcessError" on a non zero exit code
    )


def preprocess_all(config: VoiceConfig) -> list[Path]:
    """this preprocesses all the raw audio files

    It returns a list of preprocessed paths

    It iterates over all the 10 test queries and runs each through the ffmpeg pipeline

    Parameters:
    config : VoiceConfig
    the configuration object with the project's directory structure

    Returns:
    list[Path]
    the paths to all available preprocessed audio files
    """
    config.ensure_dirs()

    paths: list[Path] = []

    for query in ALL_QUERIES:
        raw = config.raw_audio_path(query.query_id)
        preprocessed = config.preprocessed_audio_path(query.query_id)

        if not raw.exists():
            print(f" Skip {query.query_id}: raw audio not found")
            continue

        if preprocessed.exists():
            print(f" Exists {query.query_id}: {preprocessed.name}")
            paths.append(preprocessed)
            continue

        print(f" Preprocessing {query.query_id}...")
        preprocess_audio(raw, preprocessed)
        paths.append(preprocessed)

    return paths
