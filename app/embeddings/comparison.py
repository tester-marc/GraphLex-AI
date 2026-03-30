"""
This is the harness for the embedding comparison: it embeds the corpus and evaluates the retrieval quality.

The GraphLex AI system ingests Swiss & EU data protection regulations (GDPR, FADP, as well as
guidance documents) and lets the users ask natural language questions about them.
This file performs benchmarks on multiple embedding models in order to pick the best one for
use in production.

This comparison works as follows:
  1. It loads all 817 document chunks (which are pre-extracted through PyMuPDF in the Ingestion Layer)
  2. For every embedding model which is tested:
     a. It embeds the whole corpus (all 817 chunks) or loads the cached embeddings
     b. For each of the 12 test queries (which are spread across 4 categories in terms of difficulty):
        - It embeds the query text
        - It ranks all 817 chunks via cosine similarity to the query vector
        - It compares this ranking against manually curated labels of relevance
        - It then computes Precision@5, Precision@10, and the Mean Reciprocal Rank (MRR)
  3. It then aggregates the per model averages and the breakdowns per category
  4. Finally it saves everything to JSON for reporting

Models that are compared:
  - OpenAI text-embedding-3-small (1536d, 512d, 256d via MRL dimension reduction)
  - OpenAI text-embedding-3-large (3072d)
  - Isaacus kanon-2-embedder (1792d, legal domain trained)

The result: "text-embedding-3-large" was chosen for production because of the best aggregate P@5 (0.600)
and MRR (0.892) at latency 5.8x lower than kanon
"""

# import libraries
from __future__ import annotations  # to make the type hints string based
import json  # for reading and writing JSON files
import math  # for using math.sqrt in the cosine similarity calculation
import re  # for regular expressions
from pathlib import Path  # for file and directory references

# project internal imports
# EMBEDDING_MODELS: the list of the 5 ModelConfig objects for which embedding models to compare
# TEST_QUERIES: for the list of 12 TestQuery objects
# EmbeddingConfig: the configuration class that provides the file paths
from app.embeddings.config import EMBEDDING_MODELS, TEST_QUERIES, EmbeddingConfig

# precision_at_k: this returns top-k results that are relevant provided a ranked list of chunk IDs and set of relevant IDs
# reciprocal_rank: returns 1/rank of first result that is relevant
from app.embeddings.metrics import precision_at_k, reciprocal_rank

# for the data model classes for structuring the results of comparison
from app.embeddings.models import (
    EmbeddingComparisonResult,
    EmbeddingResult,
    ModelConfig,
    RelevanceRule,
    TestQuery,
)


class EmbeddingComparisonHarness:
    """
    This runs the full comparison for the embedding models

    It loads the document corpus (i.e., the 817 chunks from the 6 regulatory PDFs used as documents),
    then evaluates the 12 test queries per model, and then calculates P@5, P@10, MRR and saves the results to JSON

    Used from the CLI as:
    python -m app.embeddings --compare
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        This initializes the harness with a configuration object.

        Parameters:
        config : EmbeddingConfig

        This provides the file system paths (where to find the chunks directory, the embedding cache,
        and output the results).
        """
        # stores the config
        self.config = config

        # parallel arrays, these 3 lists will be populated by load_corpus() which is defined below
        # index i refers to the same chunk across all 3 lists
        self.chunks: list[dict] = []
        self.chunk_ids: list[str] = []  # e.g., "gdpr:42"
        self.chunk_texts: list[str] = []  # the text content sent to the embedding APIs

    # Loading the corpus
    def load_corpus(self) -> None:
        """
        This loads all the PyMuPDF chunks from disk into memory

        The chunks are on disk at: data/output/pymupdf/{source_id}/chunks.json
        After the loading, self.chunks has 817 entries and the parallel
        chunk_ids / chunk_texts arrays are populated and the same length
        """
        # resets to empty in case that load_corpus is called more than one time
        self.chunks = []

        # for loop that iterates over each source dir in data/output/pymupdf/
        for source_dir in sorted(self.config.chunks_dir.iterdir()):
            # looks for chunks.json file in source dir
            chunks_file = source_dir / "chunks.json"
            if not chunks_file.exists():
                # skips the directories that don't contain a chunks.json
                continue

            # reads and parses the JSON file
            with open(chunks_file, encoding="utf-8") as f:
                doc_chunks = json.load(f)

            # appends the chunks of this document to the master list
            self.chunks.extend(doc_chunks)

        # builds the parallel arrays
        # the chunk IDs used for ranking and matching of relevance
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        # the actual texts sent to the embedding APIs
        self.chunk_texts = [c["text"] for c in self.chunks]

        # In order to print a status msg to show how many chunks were loaded
        print(
            f" Loaded {len(self.chunks)} chunks from {len(list(self.config.chunks_dir.iterdir()))} documents "
        )

    # Labelling in terms of relevance:
    #
    # The relevance is determined by the regex rules against the chunk metadata (rather than
    # manual labelling all the 817 chunks × 12 queries (which are 9,804 judgements)
    # The rules are then ordered with the highest relevance first and the first match wins

    @staticmethod
    def _matches_rule(chunk: dict, rule: RelevanceRule) -> bool:
        """
        This is a check to see if a single chunk matches a single rule of relevance

        Parameters:
        chunk: dict
        The chunk dictionary along with keys like "source_id", "article", "text", and so forth
        rule: RelevanceRule
        This specifies the source_id (optional), the match_field, and the regex pattern

        Returns:
        a bool which is true if the chunk matches this rule and otherwise false
        """
        # checks if the rule requires a specific document source and
        # immediately rejects chunks from other docs
        if rule.source_id and chunk.get("source_id") != rule.source_id:
            return False

        # extracts the value of the field that needs to be checked
        value = chunk.get(rule.match_field) or ""

        # some of the fields (e.g., cross references) are stored as lists
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)

        # applies the regex pattern
        # if match is found, returns true, otherwise false
        return bool(re.search(rule.pattern, str(value)))

    def label_relevance(self, query: TestQuery) -> dict[str, int]:
        """
        This function determines which chunks are relevant for a given test query

        Parameters:
        query: TestQuery

        Returns:
        dict[str, int]
        chunk_id is mapped to a relevance level (2 = highly relevant, 1 = relevant)
        And chunks with no matching rule are left out (level 0: not relevant)
        """
        # dict to collect the relevance labels
        labels: dict[str, int] = {}

        # this checks each chunk in the corpus against the rules of the query
        for chunk in self.chunks:
            cid = chunk["chunk_id"]
            # tries each rule in order
            for rule in query.relevance_rules:
                if self._matches_rule(chunk, rule):
                    # keeps highest level if the rules overlap
                    labels[cid] = max(labels.get(cid, 0), rule.level)
                    break

        return labels

    # Embedding and Caching:
    #
    # The corpus embeddings are cached to disk (data/output/embeddings/cache/
    # {model_label}.json) in order to avoid repeated API calls that could be expensive
    # The cache is invalidated if chunk count or the vector dimensions change
    # The query embeddings are not cached and because their latency is one of the metrics

    def _get_embedder(self, model: ModelConfig):
        """
        This function creates and returns the appropriate embedder for a model

        It supports the providers "openai" (text-embedding-3-small / large, with
        optional MRL dimension reduction) and "isaacus" (kanon-2-embedder,
        which is a legal domain model that is trained on European legal texts)

        Raises:
        ValueError
        if the provider string isn't recognised
        """
        if model.provider == "openai":
            # this avoids errors if the OpenAI SDK is not installed
            from app.embeddings.openai_embedder import OpenAIEmbedder

            return OpenAIEmbedder(model=model.name, dimensions=model.dimensions)
        elif model.provider == "isaacus":
            # like above, this is a lazy import for the kanon embedder
            from app.embeddings.kanon_embedder import KanonEmbedder

            return KanonEmbedder(dimensions=model.dimensions)

        raise ValueError(f"Unknown provider: {model.provider}")

    def _load_cache(self, model: ModelConfig) -> dict | None:
        """
        This function tries to load the cached corpus embeddings from disk

        It returns None if cache is missing, chunk count has changed,
        or if vector dimensions do not match model config
        """
        # constructs the path for the cache file
        cache_path = self.config.cache_path(model.label)

        # if the cache file does not exist there is also nothing to be loaded
        if not cache_path.exists():
            return None

        # this reads and parses the cached JSON
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)

        # this validates that the cache matches current corpus size
        if cached.get("chunk_count") != len(self.chunks):
            print(f" Mismatching chunk count, re-embedding...")
            return None

        # checks that the dimensions match
        if cached.get("dimensions") != model.dimensions:
            return None

        return cached

    def _save_cache(
        self,
        model: ModelConfig,
        embeddings: list[list[float]],
        tokens: int,
        latency_ms: float,
    ) -> None:
        """
        This function saves the corpus embeddings to disk

        Parameters:
        model: ModelConfig
        embeddings: list[list[float]]
        This is the 817 x D embedding matrix
        tokens: int
        The tokens consumed in total
        latency_ms: float
        The total time for API calls for the embedding
        """
        # builds the data structure for the cache
        cache_data = {
            "model": model.name,  # API model name
            "label": model.label,  # the label to be displayed
            "dimensions": model.dimensions,  # vector dimensionality
            "chunk_count": len(self.chunks),  # number of chunks
            "total_tokens": tokens,  # total tokens used
            "embed_latency_ms": round(latency_ms, 1),  # time taken
            "embeddings": embeddings,  # this is for the actual vectors
        }

        # writes as JSON to disk
        cache_path = self.config.cache_path(model.label)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        # to print the cache file size
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f" Cached to {cache_path.name} ({size_mb:.1f} MB) ")

    def embed_corpus(self, model: ModelConfig) -> tuple[list[list[float]], int, float]:
        """
        This function embeds the whole document corpus

        Returns:
        a tuple of (embeddings, tokens, and latency_ms)
        """
        # tries to load the cache first
        cached = self._load_cache(model)
        if cached:
            print(f" Using the cached embeddings ({cached['chunk_count']} chunks) ")
            return (
                cached["embeddings"],
                cached["total_tokens"],
                cached["embed_latency_ms"],
            )

        # if no valid cache the API needs to be called, creates the embedder instance
        embedder = self._get_embedder(model)

        # check if API key is available
        if not embedder.is_available():
            raise RuntimeError(f"The API key is not available for {model.provider}")

        # embeds all chunk texts and returns embedding matrix
        print(f" Embedding {len(self.chunk_texts)} chunks with {model.label}... ")
        embeddings, tokens, latency_ms = embedder.embed_batch(self.chunk_texts)

        # caches the results
        self._save_cache(model, embeddings, tokens, latency_ms)

        return embeddings, tokens, latency_ms

    # Similarity and Ranking:
    #
    # The cosine similarity is calculated here.

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        This function calculates cosine similarity: (a · b) / (||a|| x ||b||).

        It returns 0.0 if either of vectors is all zeros
        """
        # dot product
        dot = sum(x * y for x, y in zip(a, b))
        # L2 norms, which is the length of each vector in Euclidean space
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        # protects against dividing by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0

        # cosine similarity formula
        return dot / (norm_a * norm_b)

    def rank_chunks(
        self,
        query_embedding: list[float],
        corpus_embeddings: list[list[float]],
    ) -> list[str]:
        """
        This function ranks all the corpus chunks by cosine similarity to a query vector

        Returns:
        list[str]
        The chunk IDs sorted by similarity in descending order
        """
        # calculates the cosine similarity between the query and every chunk
        # with chunk_id and similarity_score in list of tuples
        similarities = [
            (self.chunk_ids[i], self._cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(corpus_embeddings)
        ]

        # sorts by similarity score with the hightest similarity first
        similarities.sort(key=lambda x: x[1], reverse=True)

        # returns the chunk IDs for the ranked order
        return [cid for cid, _ in similarities]

    # main loop for the comparison

    def run(self, models: list[ModelConfig] | None = None) -> list[EmbeddingResult]:
        """
        This function runs the full embedding comparison across all the models and queries

        For every model, it embeds the corpus, then labels the relevance for each of
        the 12 test queries, embeds the query, then ranks chunks, and finally computes the
        P@5, P@10, and Reciprocal Rank.

        Parameters:
        models: list[ModelConfig] or None
        This is for the models to be compared. The default is EMBEDDING_MODELS from config.py

        Returns:
        list[EmbeddingResult]
        """
        # in order to load the corpus if it hasn't been loaded
        if not self.chunks:
            self.load_corpus()

        # to create the output dirs
        self.config.ensure_dirs()

        # uses default model list
        models = models or EMBEDDING_MODELS
        # uses all 12 test queries
        queries = TEST_QUERIES
        # all individual results for all models and queries
        all_results: list[EmbeddingResult] = []

        # iterates over each embedding model
        for model in models:
            # creates an embedder to check for API key available
            embedder = self._get_embedder(model)
            if not embedder.is_available():
                # skips models where the API keys aren't configured
                print(f"\n Skipping {model.label} - API key not set ")
                continue

            # print header for results
            print(f"\n  - {model.label} ({model.dimensions}d) -")

            # embeds full corpus with this model
            corpus_embeddings, corpus_tokens, corpus_ms = self.embed_corpus(model)

            # evaluates each test query
            for query in queries:
                # in order to determine which chunks are correct answers for this query
                relevance = self.label_relevance(query)

                # hits are "highly relevant" (2) as well as "relevant" (1)
                relevant_ids = set(relevance.keys())

                # prints a warning if no chunks match against any relevance rule
                if not relevant_ids:
                    print(
                        f" Warning: No relevant chunks found for query {query.query_id}"
                    )

                # embeds the qquery text
                query_emb, q_tokens, q_ms = embedder.embed_query(query.text)

                # this ranks all chunks by cosine similarity to the query with the most similar first
                ranked = self.rank_chunks(query_emb, corpus_embeddings)

                # what fraction of the top 5 returned chunks are relevant
                p5 = precision_at_k(ranked, relevant_ids, 5)
                # what fraction of the top 10 returned chunks are relevant
                p10 = precision_at_k(ranked, relevant_ids, 10)
                # reciprocal rank, which is 1 divided by the position of the first relevant result
                rr = reciprocal_rank(ranked, relevant_ids)

                # this calculates the USD cost of embedding this query
                cost = q_tokens * model.cost_per_million_tokens / 1_000_000

                # puts everything into a dataclass
                result = EmbeddingResult(
                    model_label=model.label,  # the model label
                    dimensions=model.dimensions,  # model dimensions
                    query_id=query.query_id,  # the query id
                    query_category=query.category,  # e.g., "article_specific"
                    precision_at_5=round(p5, 4),  # round to 4 decimal places
                    precision_at_10=round(p10, 4),
                    reciprocal_rank=round(rr, 4),
                    query_latency_ms=round(q_ms, 1),  # milliseconds to 1 decimal place
                    query_tokens=q_tokens,  # token count
                    cost_usd=cost,  # cost in USD for this query
                )
                all_results.append(result)

                # prints a summary on a line for each query
                print(
                    f" {query.query_id}: P@5={p5:.2f}  P@10={p10:.2f}  MRR={rr:.2f}  ({q_ms:.0f}ms)"
                )

        return all_results

    # Aggregation: to group the results by model and calculate averages and breakdowns per category

    @staticmethod
    def _avg(values: list[float]) -> float:
        """Calculates the arithmetic mean and returns 0.0 for an empty list in order to avoid division by zero."""
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def aggregate(
        results: list[EmbeddingResult],
        corpus_stats: dict[str, tuple[int, float]] | None = None,
    ) -> list[EmbeddingComparisonResult]:
        """
        This function groups individual results by model and calculates the aggregate stats.

        It computes the overall averages (P@5, P@10, MRR, latency, cost) as well as a
        breakdown per category across the four different query categories. This goes in Tables 8
        and 9 in the final report.

        Parameters:
        results: list[EmbeddingResult]
        corpus_stats: dict[str, tuple[int, float]] or None
        This maps the model_label to (total_tokens, total_latency_ms) for the corpus
        embedding (passed in by the CLI).

        Returns:
        list[EmbeddingComparisonResult]
        One entry per model which is sorted alphabetically by label.
        """
        # this groups results by model label
        from collections import defaultdict

        groups: dict[str, list[EmbeddingResult]] = defaultdict(list)
        for r in results:
            groups[r.model_label].append(r)

        # helper to calculate the arithmetic mean of a list
        avg = lambda vals: sum(vals) / len(vals) if vals else 0.0

        # for the final aggregated results
        aggregated: list[EmbeddingComparisonResult] = []

        # to process the results of each model
        for label, items in sorted(groups.items()):

            # breakdown per category
            cats: dict[str, list[EmbeddingResult]] = defaultdict(list)
            for r in items:
                cats[r.query_category].append(r)

            # calculates per category averages for P@5, P@10, and MRR
            per_cat = {}
            for cat, cat_items in sorted(cats.items()):
                per_cat[cat] = {
                    "precision_at_5": round(
                        avg([r.precision_at_5 for r in cat_items]), 4
                    ),
                    "precision_at_10": round(
                        avg([r.precision_at_10 for r in cat_items]), 4
                    ),
                    "mrr": round(avg([r.reciprocal_rank for r in cat_items]), 4),
                }

            # stats for corpus embeddings
            corpus_tokens, corpus_ms = 0, 0.0
            if corpus_stats and label in corpus_stats:
                corpus_tokens, corpus_ms = corpus_stats[label]

            # this builds the aggregated result for this model
            aggregated.append(
                EmbeddingComparisonResult(
                    model_label=label,  # the model label
                    dimensions=items[0].dimensions,  # model dimensions
                    avg_precision_at_5=round(
                        avg([r.precision_at_5 for r in items]), 4
                    ),  # overall P@5 average
                    avg_precision_at_10=round(
                        avg([r.precision_at_10 for r in items]), 4
                    ),  # overall P@10 average
                    avg_mrr=round(
                        avg([r.reciprocal_rank for r in items]), 4
                    ),  # overall MRR average
                    avg_query_latency_ms=round(
                        avg([r.query_latency_ms for r in items]), 1
                    ),  # average ms per query
                    total_corpus_tokens=corpus_tokens,  # tokens to embed all the chunks
                    corpus_embed_latency_ms=round(
                        corpus_ms, 1
                    ),  # milliseconds to embed all the chunks
                    avg_cost_per_query=round(
                        avg([r.cost_usd for r in items]), 8
                    ),  # average USD per query
                    per_category=per_cat,  # dict of category
                    individual_results=items,  # this keeps the 12 individual results
                )
            )

        return aggregated

    # Output

    def save_results(
        self,
        results: list[EmbeddingResult],
        aggregated: list[EmbeddingComparisonResult],
    ) -> Path:
        """
        This function saves all the results of the comparison to data/output/embeddings/comparison_results.json

        Returns:
        Path
        This is the path to the JSON file saved.
        """
        # builds the output dict
        output = {
            # individual results, i.e., one entry per combination of model and query
            "individual": [
                {
                    "model_label": r.model_label,
                    "dimensions": r.dimensions,
                    "query_id": r.query_id,
                    "query_category": r.query_category,
                    "precision_at_5": r.precision_at_5,
                    "precision_at_10": r.precision_at_10,
                    "reciprocal_rank": r.reciprocal_rank,
                    "query_latency_ms": r.query_latency_ms,
                    "query_tokens": r.query_tokens,
                    "cost_usd": r.cost_usd,
                }
                for r in results
            ],
            # aggregated results, i.e., one entry per model with averages
            "aggregated": [
                {
                    "model_label": a.model_label,
                    "dimensions": a.dimensions,
                    "avg_precision_at_5": a.avg_precision_at_5,
                    "avg_precision_at_10": a.avg_precision_at_10,
                    "avg_mrr": a.avg_mrr,
                    "avg_query_latency_ms": a.avg_query_latency_ms,
                    "total_corpus_tokens": a.total_corpus_tokens,
                    "corpus_embed_latency_ms": a.corpus_embed_latency_ms,
                    "avg_cost_per_query": a.avg_cost_per_query,
                    "per_category": a.per_category,
                }
                for a in aggregated
            ],
        }

        # this writes to disk and ensure_ascii=False preserves Unicode (e.g., "§", and accented Swiss-German terms)
        path = self.config.results_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def format_results(aggregated: list[EmbeddingComparisonResult]) -> str:
        """
        This function formats the aggregated results as an ASCII table.

        And prints the overall comparison table (P@5, P@10, MRR, latency, cost)
        followed by a P@5 breakdown per category

        Returns:
        str
        A multi-line string which is ready for being printed to the terminal
        """
        # builds the header row
        header = f"{'Model':<30} {'Dims':>5} {'P@5':>6} {'P@10':>6} {'MRR':>6} {'Lat(ms)':>8} {'$/query':>10}"
        # with a row of dashes
        sep = "-" * len(header)

        # builds the output lines
        lines = [sep, header, sep]

        # adds a row per model
        for a in aggregated:
            lines.append(
                f"{a.model_label:<30} {a.dimensions:>5} "
                f"{a.avg_precision_at_5:>6.3f} {a.avg_precision_at_10:>6.3f} "
                f"{a.avg_mrr:>6.3f} {a.avg_query_latency_ms:>8.1f} "
                f"${a.avg_cost_per_query:>9.6f}"
            )

        # a bottom separator
        lines.append(sep)

        # breakdown table per category
        lines.append("\nPer category Precision@5:")

        # this collects all unique category names
        cats = sorted({cat for a in aggregated for cat in a.per_category})

        # builds the category header row
        cat_header = f"{'Model':<30} " + " ".join(f"{c[:12]:>13}" for c in cats)
        lines.append(cat_header)

        # adds a row per model
        for a in aggregated:
            vals = " ".join(
                f"{a.per_category.get(c, {}).get('precision_at_5', 0):>13.3f}"
                for c in cats
            )
            lines.append(f"{a.model_label:<30} {vals}")

        # joins all the lines with newlines
        return "\n".join(lines)
