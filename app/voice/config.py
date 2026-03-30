"""this is for the configuration for the voice input comparison

For the GraphLex AI voice input comparison pipeline (Layer 2):
1. which transcription models to test (Whisper sizes + Voxtral)
2. what regulatory vocabulary to use for context biasing
3. the 10 test queries (with the ground truth text and regulatory entities)
4. where the audio files, preprocessed files, and the results are stored on disk

all the other modules (audio_generator.py, comparison.py, whisper_transcriber.py, etc.)
import from here
"""

# import libraries
from __future__ import annotations
from pathlib import Path
from app.voice.models import TestQuery


# Model configuration

# these are the Whisper sizes to evaluate
WHISPER_MODELS = ["tiny", "base", "small", "medium"]

# Voxtral Mini (4.7B params) via the Mistral API (it is in the cloud, not local)
VOXTRAL_MODEL = "mistral-small-latest"  # Voxtral Mini via audio transcription


# Vocab for Context biasing

# Whisper accepts a free text "initial_prompt" and Voxtral accepts a single-token list

# the master list of domain specific regulatory terms (single words and phrases)
# which is used to construct WHISPER_CONTEXT_PROMPT
REGULATORY_VOCABULARY = [
    # acronyms
    "FADP",
    "GDPR",
    "EDPB",
    "FDPIC",
    "DPO",
    "DPA",
    # legal document structure terms
    "Article",
    "Recital",
    "Annex",
    "Chapter",
    "Section",
    "Paragraph",
    # actors in data protection law
    "data controller",
    "data processor",
    "data subject",
    # main legal concepts
    # "legitimate interest" = lawful basis under GDPR Art.6
    # "adequacy decision" = EU Commission ruling that a non-EU country provides adequate protection
    "legitimate interest",
    "consent",
    "adequacy decision",
    # cross-border and breach related terms
    "cross-border transfer",
    "data breach notification",
    # the British spelling "organisational" matches the official GDPR text
    "technical and organisational measures",
    # GDPR Art.9 term for sensitive data categories
    "special categories of personal data",
    # the full names of key roles and bodies
    "Data Protection Officer",
    "Data Processing Agreement",
    "Swiss Federal Act on Data Protection",
    "European Data Protection Board",
    "Federal Data Protection and Information Commissioner",
    # additional regulatory concepts
    "supervisory authority",
    "binding corporate rules",
    "data protection impact assessment",
    "DPIA",
    # data subject rights
    "right to erasure",
    "right of access",
    "data portability",
    # terms related to processing
    "lawful basis",
    "processing activity",
    "records of processing",
]

# Voxtral only accepts single tokens
# therefore multi-word phrases are reduced to their most distinctive word
VOXTRAL_CONTEXT_BIAS = [
    # acronyms
    "FADP",
    "GDPR",
    "EDPB",
    "FDPIC",
    "DPO",
    "DPA",
    "DPIA",
    # document structure terms
    "Article",
    "Recital",
    "Annex",
    "Chapter",
    "Section",
    "Paragraph",
    # distinctive single words extracted from multi-word phrases
    "controller",
    "processor",
    "consent",
    "adequacy",
    "supervisory",
    "cross-border",
    "notification",
    "organisational",
    "portability",
    "erasure",
    "lawful",
    "processing",
    "Ordinance",
    "Commissioner",
    # additional single word regulatory terms
    "recitals",
    "provisions",
    "obligations",
    "jurisdiction",
    "derogation",
    "safeguards",
    "legitimate",
    "proportionality",
]

# this is passed to Whisper as "initial_prompt" to bias transcription towards the regulatory vocabulary
WHISPER_CONTEXT_PROMPT = (
    "This is a regulatory compliance query about Swiss and EU data protection law. "
    "Key terms: FADP, GDPR, EDPB, FDPIC, Article, Recital, Annex, "
    "data controller, data processor, data subject, legitimate interest, "
    "adequacy decision, cross-border transfer, DPIA, DPO."
)


# Test queries for Category A: General regulatory speech

# conversational regulatory questions with some domain terms but no specific article references
# this tests recognition of regulatory vocab in natural speech
CATEGORY_A_QUERIES = [
    TestQuery(
        query_id="a1",
        category="A",
        ground_truth="What are the requirements for data processing under the new Swiss data protection law?",
        regulatory_entities=["Swiss", "data protection"],
    ),
    TestQuery(
        query_id="a2",
        category="A",
        # "transferring personal data to third countries" is a precise legal phrase
        ground_truth="Can you explain the legal basis for transferring personal data to third countries?",
        regulatory_entities=["personal data", "third countries"],
    ),
    TestQuery(
        query_id="a3",
        category="A",
        ground_truth="What obligations does our company have regarding data breach notification?",
        regulatory_entities=["data breach notification"],
    ),
    TestQuery(
        query_id="a4",
        category="A",
        ground_truth="How should we handle consent withdrawal in our customer management system?",
        regulatory_entities=["consent"],
    ),
    TestQuery(
        query_id="a5",
        category="A",
        # "technical and organisational measures" is an exact GDPR phrase
        ground_truth="What technical and organisational measures are required to protect personal data?",
        regulatory_entities=["technical and organisational measures", "personal data"],
    ),
]


# Test queries for Category B: Regulatory entity dense speech

# these are dense clusters of specific article numbers, paragraph references, acronyms, and cross-references
# a single transcription error (e.g., "Article 16" -> "Article 60") could route to the wrong provision
CATEGORY_B_QUERIES = [
    TestQuery(
        query_id="b1",
        category="B",
        # cross-jurisdictional: FADP Art.6(3) vs. GDPR Art.6(1)(f)
        ground_truth="Does FADP Article 6 paragraph 3 align with GDPR Article 6 paragraph 1 point f on legitimate interest?",
        regulatory_entities=[
            "FADP",
            "Article 6",
            "paragraph 3",
            "GDPR",
            "Article 6",
            "paragraph 1",
            "point f",
            "legitimate interest",
        ],
    ),
    TestQuery(
        query_id="b2",
        category="B",
        ground_truth="Under EDPB Guidelines 2/2018, what conditions must be met for Article 48 data transfers?",
        regulatory_entities=[
            "EDPB",
            "Guidelines 2/2018",
            "Article 48",
            "data transfers",
        ],
    ),
    TestQuery(
        query_id="b3",
        category="B",
        # cross-jurisdictional: GDPR Art.49(1)(a) vs. FADP Art.16
        ground_truth="How does Article 49 paragraph 1 point a of the GDPR interact with FADP Article 16?",
        regulatory_entities=[
            "Article 49",
            "paragraph 1",
            "point a",
            "GDPR",
            "FADP",
            "Article 16",
        ],
    ),
    TestQuery(
        query_id="b4",
        category="B",
        # FDPIC = Swiss federal data protection authority, Data Protection Ordinance = this is secondary legislation to FADP
        ground_truth="What are the FDPIC requirements under Article 8 of the Data Protection Ordinance?",
        regulatory_entities=[
            "FDPIC",
            "Article 8",
            "Data Protection Ordinance",
        ],
    ),
    TestQuery(
        query_id="b5",
        category="B",
        # recitals are preamble paragraphs in EU legislation that provide interpretive context
        ground_truth="Compare GDPR Recital 47 on legitimate interest with the EDPB guidelines on balancing tests in Section 3.",
        regulatory_entities=[
            "GDPR",
            "Recital 47",
            "legitimate interest",
            "EDPB",
            "Section 3",
            "balancing tests",
        ],
    ),
]

# a combined list used by audio_generator.py, preprocessing.py, and comparison.py
ALL_QUERIES = CATEGORY_A_QUERIES + CATEGORY_B_QUERIES


# VoiceConfig: file path mgmt


class VoiceConfig:
    """paths and settings for the voice comparison pipeline

    this centralizes all file system path logic so no module hardcodes the paths
    """

    def __init__(self, project_root: Path | None = None):
        """this initializes with optional project root override

        Args:
        project_root: the root dir of the GraphLex AI project
        it is automatically detected from this file's location if not provided
        """
        self.project_root = project_root or Path(__file__).resolve().parents[2]

        self.audio_dir = self.project_root / "data" / "audio"
        self.raw_dir = self.audio_dir / "raw"  # TTS output (audio_generator.py)
        self.preprocessed_dir = (
            self.audio_dir / "preprocessed"
        )  # ffmpeg output (preprocessing.py)
        self.output_dir = self.project_root / "data" / "output" / "voice"

    def ensure_dirs(self) -> None:
        """this creates all necessary dirs if they don't already exist"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def raw_audio_path(self, query_id: str) -> Path:
        """the Path for a raw (preprocessing) audio file"""
        return self.raw_dir / f"{query_id}.wav"

    def preprocessed_audio_path(self, query_id: str) -> Path:
        """the Path for a preprocessed audio file

        ffmpeg filters applied are: loudnorm (-16 LUFS), high-pass at 80Hz, 16kHz mono 16bit PCM

        """
        return self.preprocessed_dir / f"{query_id}.wav"

    def results_path(self) -> Path:
        """the Path for the comparison results JSON file

        JSON structure:
            {
                "individual": [ ... ],  // one entry per transcription attempt
                "aggregated": [ ... ]   // one entry per model + config combo
            }
        """
        return self.output_dir / "comparison_results.json"
