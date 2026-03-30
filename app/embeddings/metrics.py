"""This script is for the retrieval evaluation metrics (Precision@k and Mean Reciprocal Rank (MRR))"""

# It contains 2 standard Information Retrieval metrics used to score the ranking for the embedding model:
# 1. Precision@k: the fraction of the top-k results that are relevant
# 2. Reciprocal Rank: the inverse rank of the first relevant result
#
# The script is called from comparison.py for every model/query pair and
# then averaged across the 12 test queries in order to produce P@5, P@10, and MRR
# (the results have been inserted in the final report in Tables 8 and 9).

# import libraries
from __future__ import annotations  # this allows for type hint syntax


def precision_at_k(
    ranked_chunk_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """For the fraction of the top-k retrieved chunks that are relevant (i.e., level >= 1).

    Formula: P@k = (relevant items in top k) / k
    It is called twice per model/query pair, with k=5 and k=10
    - ranked_chunk_ids: the 817 chunk IDs sorted by cosine similarity in descending order
    - relevant_ids: the chunks that are prejudged as relevant (this is defined via the regex in config.py)
    """

    # slices to the top-k results
    top_k = ranked_chunk_ids[:k]

    # edge case for empty corpus
    if not top_k:
        return 0.0

    # this counts the relevant hits
    hits = sum(1 for cid in top_k if cid in relevant_ids)

    # divides by k, not len(top_k)
    return hits / k


def reciprocal_rank(
    ranked_chunk_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """1 / rank of the first relevant result (this is 0.0 if none are found in list)

    RR = 1 / (1-indexed position of the first relevant result)
    It is averaged across the queries to produce the Mean Reciprocal Rank (MRR)
    """

    for i, cid in enumerate(ranked_chunk_ids):
        if cid in relevant_ids:
            # this returns the reciprocal of 1-based rank (only first hit matters)
            return 1.0 / (i + 1)

    # when no relevant chunk was found anywhere in the ranking
    return 0.0
