"""this is for the data models for the voice input comparison

there are 3 dataclasses that represent the full lifecycle of the voice transcription evaluation:

1. TestQuery: a pre-defined test sentence with known correct text
2. TranscriptionResult: the output of a single model that transcribes 1 audio clip, with the accuracy scores
3. VoiceComparisonResult: the averaged scores across all the queries for one combo of model and configuration

It is used throughout the pipeline: "config.py" creates TestQuerys, "comparison.py" creates
TranscriptionResults and VoiceComparisonResults which populate the tables in the final report
"""

# import libraries
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TestQuery:
    """for a test query with known ground truth text

    the 10 queries are split into 2 categories:
    - Category A (5 queries): general regulatory speech
    - Category B (5 queries): entity dense speech with acronyms (e.g., FADP, GDPR),
      article numbers, and legal terminology. these should be harder to transcribe
    """

    # a short unique ID, e.g., "a1", "b3", etc.
    query_id: str

    category: str  # "A" (general) or "B" (entity dense)

    # the exact spoken text
    ground_truth: str

    # the regulatory terms (e.g., ["FADP", "Article 6", "GDPR"]) used by the "entity_wer"
    # metric in "metrics.py" in order to compute error rate on the legal terminology
    regulatory_entities: list[str] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """this is the output from a single transcription attempt

    it stores everything about one model that transcribes one audio clip:
    model, config, raw output, and accuracy scores
    the full comparison produces up to 5 models x 2 pre-processing modes x 2 biasing settings
    x 10 queries = 200 of these objects in total
    """

    # "whisper-tiny/base/small/medium" or "voxtral-mini"
    model_name: str

    query_id: str

    # this is copied from TestQuery for convenience
    category: str

    # this is copied from TestQuery to make each result self contained
    ground_truth: str

    # the raw text output from the speech model
    transcription: str

    # "raw": original TTS WAV that is fed directly to the model
    # "ffmpeg": the audio normalized to 16kHz mono WAV and noise filtered
    preprocessing: str

    # true: the model received regulatory vocabulary hints (Whisper: initial_prompt,
    # Voxtral: context_bias)
    # false: general purpose recognition
    context_biasing: bool

    # time from model call to the response, in ms
    latency_ms: float

    # the fraction of all the words that were wrong (0.0 is perfect, 1.0 is completely wrong)
    general_wer: float = 0.0

    # the fraction of regulatory terms that were wrong
    entity_wer: float = 0.0

    # the count of hallucinated words not present in the audio
    fabricated_insertion_count: int = 0


@dataclass
class VoiceComparisonResult:
    """for the aggregate results for one model + preprocessing + biasing config

    this averages the scores across all the 10 test queries in order to produce a single summary row
    for the comparison tables in the report
    """

    model_name: str
    preprocessing: str  # "raw" or "ffmpeg"
    context_biasing: bool

    # the averages across all the 10 queries
    avg_general_wer: float = 0.0
    avg_entity_wer: float = 0.0
    avg_fic: float = 0.0  # can be fractional
    avg_latency_ms: float = 0.0

    # breakdowns per category
    cat_a_wer: float = 0.0
    cat_b_wer: float = 0.0
    cat_b_entity_wer: float = 0.0  # this is only computed for Cat B

    # individual results are kept for traceability
    individual_results: list[TranscriptionResult] = field(default_factory=list)
