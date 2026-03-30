"""this is the voice comparison harness, it runs all transcribers, compute the metrics, and save results

It coordinates the full comparison (for Layer 2):
1. it runs Whisper tiny/base/small/medium and Voxtral Mini on all the 10 test queries
2. tests each model under 4 configs: raw / ffmpeg audio x  no / yes for context biasing
3. computes WER, entity WER, and FIC for every transcription
4. and aggregates the results by model + config and saves to JSON

It provides 5 models x 4 configs x 10 queries = 200 individual transcription results

It also provides the quantitative evidence for the model selection (which was ultimately Whisper small)
and rejections (Voxtral Mini, Whisper models tiny/base/medium) for the final report

Metrics:
WER         : (substitutions + insertions + deletions) / total reference words
Entity WER  : the WER computed only on the regulatory terms (e.g., "FADP", "Article 6")
FIC         : the count of hallucinated words that are not present in the original audio

Files:
reads:   data/audio/raw/*.wav, data/audio/preprocessed/*.wav
writes:  data/output/voice/comparison_results.json
calls:   whisper_transcriber.py, voxtral_transcriber.py, metrics.py
"""

# import libraries
from __future__ import annotations
import json
from pathlib import Path
from app.voice.config import ALL_QUERIES, WHISPER_MODELS, VoiceConfig
from app.voice.models import TestQuery, TranscriptionResult, VoiceComparisonResult
from app.voice import metrics


class VoiceComparisonHarness:
    """this runs Whisper and Voxtral on the test audio, computes the metrics, aggregates

    Usage:
    config = VoiceConfig()
    harness = VoiceComparisonHarness(config)
    whisper_results = harness.run_whisper()
    voxtral_results = harness.run_voxtral()
    all_results = whisper_results + voxtral_results
    aggregated = harness.aggregate(all_results)
    harness.save_results(all_results, aggregated)
    print(harness.format_results(aggregated))
    """

    def __init__(self, config: VoiceConfig):
        self.config = config  # for testability

    def run_whisper(self) -> list[TranscriptionResult]:
        """this runs all Whisper sizes on all the queries for all 4 configs

        Returns:
        a list of TranscriptionResult objects
        """
        # this avoids slow whisper / PyTorch import if only Voxtral is tested
        from app.voice import whisper_transcriber

        results: list[TranscriptionResult] = []

        # tiny=39M, base=74M, small=244M (the one used production), medium=769M parameters
        for model_size in WHISPER_MODELS:
            print(f"\n  Whisper {model_size} ")

            # "ffmpeg" applies loudnorm + an 80Hz high-pass filter via "preprocessing.py"
            for preprocessing in ["raw", "ffmpeg"]:

                # context biasing passes WHISPER_CONTEXT_PROMPT as initial_prompt
                for context_biasing in [False, True]:
                    bias_label = "biased" if context_biasing else "unbiased"
                    print(f" [{preprocessing}, {bias_label}]")

                    for query in ALL_QUERIES:
                        audio_path = (
                            self.config.preprocessed_audio_path(query.query_id)
                            if preprocessing == "ffmpeg"
                            else self.config.raw_audio_path(query.query_id)
                        )

                        if not audio_path.exists():
                            print(f" Skip {query.query_id}: audio not found")
                            continue

                        # the model is cached after the first load per size and runs CPU
                        transcription, latency = whisper_transcriber.transcribe(
                            audio_path, model_size, context_biasing
                        )

                        result = self._score(
                            model_name=f"whisper-{model_size}",
                            query=query,
                            transcription=transcription,
                            preprocessing=preprocessing,
                            context_biasing=context_biasing,
                            latency_ms=latency,
                        )
                        results.append(result)

                        wer_pct = result.general_wer * 100
                        print(
                            f" {query.query_id}: WER={wer_pct:.1f}% latency={latency:.0f}ms"
                        )

        return results

    def run_voxtral(self) -> list[TranscriptionResult]:
        """this run Voxtral Mini on all the queries under all 4 configs

        Voxtral is a cloud API (from Mistral)
        API calls are wrapped in try/except for possible
        network, auth or rate-limit failures

        Returns:
        a list of TranscriptionResult objects or [] if the API key is missing
        """
        from app.voice import voxtral_transcriber

        if not voxtral_transcriber.is_available():
            print(" Skip Voxtral: MISTRAL_API_KEY is not set")
            return []

        results: list[TranscriptionResult] = []
        print("\n  Voxtral Mini ")

        for preprocessing in ["raw", "ffmpeg"]:
            for context_biasing in [False, True]:
                bias_label = "biased" if context_biasing else "unbiased"
                print(f" [{preprocessing}, {bias_label}]")

                for query in ALL_QUERIES:
                    audio_path = (
                        self.config.preprocessed_audio_path(query.query_id)
                        if preprocessing == "ffmpeg"
                        else self.config.raw_audio_path(query.query_id)
                    )
                    if not audio_path.exists():
                        print(f" Skip {query.query_id}: audio not found")
                        continue

                    try:
                        # context_biasing passes the VOXTRAL_CONTEXT_BIAS token list
                        transcription, latency = voxtral_transcriber.transcribe(
                            audio_path, context_biasing
                        )
                    except Exception as e:
                        # this covers: network timeout, invalid key, rate limit, bad format
                        print(f" Error {query.query_id}: {e}")
                        continue

                    result = self._score(
                        model_name="voxtral-mini",
                        query=query,
                        transcription=transcription,
                        preprocessing=preprocessing,
                        context_biasing=context_biasing,
                        latency_ms=latency,
                    )
                    results.append(result)
                    wer_pct = result.general_wer * 100
                    print(
                        f" {query.query_id}: WER={wer_pct:.1f}% latency={latency:.0f}ms"
                    )

        return results

    def _score(
        self,
        model_name: str,
        query: TestQuery,
        transcription: str,
        preprocessing: str,
        context_biasing: bool,
        latency_ms: float,
    ) -> TranscriptionResult:
        """this computes all 3 metrics for a single transcription and returns a TranscriptionResult

        Metrics:
        general_wer   : the WER across all words
        entity_wer    : the WER on regulatory entity tokens from "query.regulatory_entities"
        fic           : the count of hallucinated words not in the ground truth
        """
        return TranscriptionResult(
            model_name=model_name,
            query_id=query.query_id,
            category=query.category,
            ground_truth=query.ground_truth,
            transcription=transcription,
            preprocessing=preprocessing,
            context_biasing=context_biasing,
            latency_ms=latency_ms,
            general_wer=metrics.general_wer(query.ground_truth, transcription),
            entity_wer=metrics.entity_wer(
                query.ground_truth, transcription, query.regulatory_entities
            ),
            fabricated_insertion_count=metrics.fabricated_insertion_count(
                query.ground_truth, transcription
            ),
        )

    @staticmethod
    def aggregate(results: list[TranscriptionResult]) -> list[VoiceComparisonResult]:
        """to group the individual results by (model, preprocessing, biasing) and average metrics

        breakdowns per category (cat_a/cat_b) show if a model struggles
        specifically with entity dense speech (Category B is the harder case)

        Returns:
        a list of VoiceComparisonResult objects sorted by the grouping key
        """
        # groups by the config tuple
        groups: dict[tuple, list[TranscriptionResult]] = {}
        for r in results:
            key = (r.model_name, r.preprocessing, r.context_biasing)
            groups.setdefault(key, []).append(r)

        aggregated: list[VoiceComparisonResult] = []

        for (model, preproc, biasing), group in sorted(groups.items()):
            cat_a = [r for r in group if r.category == "A"]
            cat_b = [r for r in group if r.category == "B"]

            aggregated.append(
                VoiceComparisonResult(
                    model_name=model,
                    preprocessing=preproc,
                    context_biasing=biasing,
                    avg_general_wer=_avg([r.general_wer for r in group]),
                    avg_entity_wer=_avg([r.entity_wer for r in group]),
                    avg_fic=_avg([r.fabricated_insertion_count for r in group]),
                    avg_latency_ms=_avg([r.latency_ms for r in group]),
                    cat_a_wer=_avg([r.general_wer for r in cat_a]),
                    cat_b_wer=_avg([r.general_wer for r in cat_b]),
                    cat_b_entity_wer=_avg([r.entity_wer for r in cat_b]),
                    # individual_results is kept for the per query drill down if needed
                    individual_results=group,
                )
            )

        return aggregated

    def save_results(
        self,
        results: list[TranscriptionResult],
        aggregated: list[VoiceComparisonResult],
    ) -> Path:
        """this saves individual and aggregated results to JSON

        numeric rounding: latency=1dp, WER/Entity WER=4dp, FIC=int/2dp

        Returns:
        the Path to the saved JSON file (data/output/voice/comparison_results.json)
        """
        output_path = self.config.results_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # dict construction to control rounding, field naming,
        # and exclusion of individual_results from the aggregated entries
        data = {
            "individual": [
                {
                    "model": r.model_name,
                    "query_id": r.query_id,
                    "category": r.category,
                    "ground_truth": r.ground_truth,
                    "transcription": r.transcription,
                    "preprocessing": r.preprocessing,
                    "context_biasing": r.context_biasing,
                    "latency_ms": round(r.latency_ms, 1),
                    "general_wer": round(r.general_wer, 4),
                    "entity_wer": round(r.entity_wer, 4),
                    "fic": r.fabricated_insertion_count,
                }
                for r in results
            ],
            "aggregated": [
                {
                    "model": a.model_name,
                    "preprocessing": a.preprocessing,
                    "context_biasing": a.context_biasing,
                    "avg_general_wer": round(a.avg_general_wer, 4),
                    "avg_entity_wer": round(a.avg_entity_wer, 4),
                    "avg_fic": round(a.avg_fic, 2),
                    "avg_latency_ms": round(a.avg_latency_ms, 1),
                    "cat_a_wer": round(a.cat_a_wer, 4),
                    "cat_b_wer": round(a.cat_b_wer, 4),
                    "cat_b_entity_wer": round(a.cat_b_entity_wer, 4),
                }
                for a in aggregated
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return output_path

    @staticmethod
    def format_results(aggregated: list[VoiceComparisonResult]) -> str:
        """in order to pretty print the aggregate results as a console table

        Columns: Model, Preproc, Bias, WER, EntWER, FIC, ms, CatA, CatB, B-Ent

        Returns:
        A multi-line table string or "No results." if the list is empty
        """
        if not aggregated:
            return "No results."

        header = (
            f"{'Model':<18} {'Preproc':<8} {'Bias':<6} "
            f"{'WER':>6} {'EntWER':>7} {'FIC':>5} {'ms':>8} "
            f"{'CatA':>6} {'CatB':>6} {'B-Ent':>6}"
        )
        lines = [header, "-" * len(header)]

        for a in aggregated:
            lines.append(
                f"{a.model_name:<18} {a.preprocessing:<8} "
                f"{'yes' if a.context_biasing else 'no':<6} "
                f"{a.avg_general_wer:>5.1%} {a.avg_entity_wer:>6.1%} "
                f"{a.avg_fic:>5.1f} {a.avg_latency_ms:>7.0f} "
                f"{a.cat_a_wer:>5.1%} {a.cat_b_wer:>5.1%} "
                f"{a.cat_b_entity_wer:>5.1%}"
            )

        return "\n".join(lines)


def _avg(values: list[float]) -> float:
    """this returns the arithmetic mean of the values or 0.0 if the list is empty"""
    return sum(values) / len(values) if values else 0.0
