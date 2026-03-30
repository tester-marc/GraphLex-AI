"""this is the CLI entry point: python -m app.voice [--generate] [--preprocess] [--compare] [--whisper-only] [--voxtral-only]

the voice comparison pipeline has three stages in sequence:

stage 1: Generate (--generate)
this uses edge-tts to create synthetic WAV files from the 10 groundtruth test queries
Output: data/audio/raw/a1.wav to b5.wav

stage 2: preprocess (--preprocess)
this runs each raw file through ffmpeg to resample to 16kHz mono and a noise filter
Output: data/audio/preprocessed/a1.wav to b5.wav

stage 3: compare (--compare, --whisper-only, --voxtral-only)
Whisper: 4 sizes x 4 configs x 10 queries, and Voxtral Mini: 4 configs x 10 queries
Metrics: WER, Entity WER, FIC, latency
Output: data/output/voice/comparison_results.json, and a printed summary

"--all" runs all 3 stages in sequence

Usage:
python -m app.voice --all           # the full pipeline: generate -> preprocess -> compare
python -m app.voice --generate      # only generate the test audio
python -m app.voice --preprocess    # only run the ffmpeg preprocessing
python -m app.voice --compare       # runs both Whisper and Voxtral comparison
python -m app.voice --whisper-only  # runs only Whisper
python -m app.voice --voxtral-only  # run only Voxtral (this needs the MISTRAL_API_KEY)
"""

# import libraries
from __future__ import annotations
import argparse
import sys
from app.voice.config import VoiceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice input comparison")

    parser.add_argument(
        "--generate", action="store_true", help="Generates test audio via TTS"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Runs ffmpeg pre-processing"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Runs the full comparison"
    )
    # --whisper-only, this is useful when the MISTRAL_API_KEY isn't set (because Whisper runs locally)
    parser.add_argument(
        "--whisper-only", action="store_true", help="Runs Whisper comparison only"
    )
    # --voxtral-only, reruns only this API based model without rerunning Whisper
    parser.add_argument(
        "--voxtral-only", action="store_true", help="Runs Voxtral comparison only"
    )
    parser.add_argument(
        "--all", action="store_true", help="generate + preprocess + compare"
    )

    # argparse converts hyphens to underscores (--whisper-only -> args.whisper_only)
    args = parser.parse_args()
    config = VoiceConfig()

    if not any(
        [
            args.generate,
            args.preprocess,
            args.compare,
            args.whisper_only,
            args.voxtral_only,
            args.all,
        ]
    ):
        parser.print_help()
        sys.exit(1)

    # lazy imports
    if args.all or args.generate:
        print("\n Generating test audio ")
        from app.voice.audio_generator import generate_test_audio

        paths = generate_test_audio(config)
        print(f" Generated {len(paths)} audio files")

    if args.all or args.preprocess:
        print("\n Preprocessing audio ")
        from app.voice.preprocessing import preprocess_all

        paths = preprocess_all(config)
        print(f" Preprocessed {len(paths)} files")

    if args.all or args.compare or args.whisper_only or args.voxtral_only:
        from app.voice.comparison import VoiceComparisonHarness

        harness = VoiceComparisonHarness(config)
        all_results = []

        if args.all or args.compare or args.whisper_only:
            print("\n Whisper comparison ")
            whisper_results = harness.run_whisper()
            all_results.extend(whisper_results)

        # "run_voxtral()" skips if the MISTRAL_API_KEY has not been set
        if args.all or args.compare or args.voxtral_only:
            print("\n Voxtral comparison ")
            voxtral_results = harness.run_voxtral()
            all_results.extend(voxtral_results)

        if all_results:
            aggregated = harness.aggregate(all_results)

            print("\n Aggregate Results ")
            print(harness.format_results(aggregated))

            # these results are used in the tables in chapters "Implementation" and "Evaluation"
            output_path = harness.save_results(all_results, aggregated)
            print(f"\n Results saved to {output_path}")


if __name__ == "__main__":
    main()
