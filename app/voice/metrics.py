"""this is for the WER, entity WER, and Fabricated Insertion Count (FIC) metrics

It evaluates how accurately speech to text models (Whisper and Voxtral) transcribe
spoken regulatory compliance queries for the GraphLex AI project

1. General WER (Word Error Rate): this is the fraction of words that are wrong overall (0.0 would be perfect)
2. Entity WER: WER that is measured only on regulatory terms (e.g., "GDPR", "Article 49")
3. Fabricated Insertion Count (FIC): the words that the model invented that weren't spoken (Wright et al., 2025)

this is used by "app/voice/comparison.py" and the results populate the tables in the project report
"""

# import libraries
from __future__ import annotations
import re
from jiwer import wer as compute_wer


def general_wer(ground_truth: str, transcription: str) -> float:
    """this is to compute the WER between ground truth and the transcription

    WER = (substitutions + insertions + deletions) / Words_in_ground_truth

    Parameters:
    ground_truth : str
    the known correct text (from "config.py" and audio generated via edge-tts)
    transcription : str
    the text produced by the speech model that is being evaluated

    Returns:
    float
    it returns the WER as a decimal
    """
    gt = _normalise(ground_truth)
    hyp = _normalise(transcription)

    if not gt:
        return 0.0 if not hyp else 1.0

    return compute_wer(gt, hyp)


def entity_wer(
    ground_truth: str,
    transcription: str,
    entities: list[str],
) -> float:
    """in order to calculate the WER only on the regulatory entity spans

    Parameters:
    ground_truth : str
    the known correct text
    transcription : str
    the model output text
    entities : list[str]
    the regulatory entities for this query (which are defined in "config.py" per query)

    Returns:
    float
    the entity WER as a decimal
    """
    if not entities:
        return 0.0

    # lowercase in order to have case insensitive matching
    gt_lower = ground_truth.lower()
    hyp_lower = transcription.lower()

    gt_entity_tokens: list[str] = []
    hyp_entity_tokens: list[str] = []

    for entity in entities:
        entity_lower = entity.lower()
        entity_tokens = entity_lower.split()

        if entity_lower in gt_lower:
            gt_entity_tokens.extend(entity_tokens)

        if entity_lower in hyp_lower:
            hyp_entity_tokens.extend(entity_tokens)
        else:
            # count the individual tokens that are present
            for token in entity_tokens:
                if token in hyp_lower:
                    hyp_entity_tokens.append(token)

    if not gt_entity_tokens:
        return 0.0

    gt_str = " ".join(gt_entity_tokens)
    hyp_str = " ".join(hyp_entity_tokens) if hyp_entity_tokens else ""

    if not hyp_str:
        return 1.0

    return compute_wer(gt_str, hyp_str)


def fabricated_insertion_count(ground_truth: str, transcription: str) -> int:
    """this is to count the fabricated insertions

    sometimes speech models hallucinate words that are never spoken, which
    can be quite harmful in a legal context.

    Parameters:
    ground_truth : str
    the known correct text of what was spoken
    transcription : str
    the output text from the model

    Returns:
    an int
    returns the number of fabricated words, lower is better
    """
    gt_words = set(_normalise(ground_truth).split())
    hyp_words = _normalise(transcription).split()

    # here are some filler words that are excluded from the fabrication counts
    fillers = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "to",
        "of",
        "in",
        "for",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "into",
        "that",
        "this",
        "it",
        "its",
        "i",
        "we",
        "you",
        "they",
        "my",
        "our",
        "your",
        "their",
        "not",
        "no",
        "yes",
    }

    fabricated = 0
    for word in hyp_words:
        if word not in gt_words and word not in fillers:
            fabricated += 1

    return fabricated


def _normalise(text: str) -> str:
    """this is to normalize text for the WER comparison, i.e.,
    lowercase, strip the punctuation, collapse whitespaces

    the dashes are preserved (e.g., "cross-border" stays)

    Parameters:
    text : str
    this is the raw text to normalize

    Returns:
    str
    the normalized text: lowercase, no punctuation, single spaced
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)  # remove the punctuation, keep the dashes
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()
