"""
this is the LLM Comparison harness

It compares 2 candidate LLMs for GraphLex AI's regulatory compliance pipeline
(GDPR, and Swiss FADP)

for each of 8 benchmark queries, it retrieves the top-5
relevant chunks from the corpus, sends them to each model, and then
evaluates citation accuracy, calibration, latency, and cost

The models compared:
- Llama 3.3 70B
- Qwen3-Next 80B-A3B (MoE, 3B active params per token)

Both models are hosted on Together AI via an API

End Result: Qwen3-Next 80B-A3B was selected for the project (for details see final report)
"""

# import libraries
from __future__ import annotations
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from app.llm.config import (
    BENCHMARK_QUERIES,
    LLM_MODELS,
    SYSTEM_PROMPT,
    UNCERTAINTY_PHRASES,
    LLMConfig,
)
from app.llm.models import (
    BenchmarkQuery,
    LLMComparisonResult,
    LLMEvaluationResult,
    LLMModelConfig,
    LLMResponse,
)
from app.llm.together_client import TogetherClient


class LLMComparisonHarness:
    """
    this orchestrates the full LLM comparison pipeline:
    1. it loads the regulatory text corpus and pre-computed embeddings
    2. retrieves top-k context chunks per benchmark query
    3. sends each query plus context to each candidate LLM
    4. evaluates the responses (citation precision/recall, calibration)
    5. and aggregates results per model and saves to JSON
    """

    TOP_K = 5  # the context chunks provided to the LLM per query

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.chunks: list[dict] = []
        self.chunk_texts: list[str] = []
        # each embedding is a list of 3,072 floats ("text-embedding-3-large")
        self.corpus_embeddings: list[list[float]] | None = None
        self.client = TogetherClient()

    # Loading of corpus and embeddings

    def load_corpus(self) -> None:
        """
        this loads all the text chunks from data/output/pymupdf/{source_id}/chunks.json
        it populates self.chunks (dicts) and self.chunk_texts (parallel strings)
        """
        self.chunks = []

        for source_dir in sorted(self.config.chunks_dir.iterdir()):
            chunks_file = source_dir / "chunks.json"
            if not chunks_file.exists():
                continue
            with open(chunks_file, encoding="utf-8") as f:
                doc_chunks = json.load(f)
            self.chunks.extend(doc_chunks)

        self.chunk_texts = [c["text"] for c in self.chunks]
        print(
            f" Loaded {len(self.chunks)} chunks from {len(list(self.config.chunks_dir.iterdir()))} documents"
        )

    def load_embeddings(self) -> None:
        """
        to load cached corpus embeddings from
        data/output/embeddings/cache/text-embedding-3-large.json

        Raises:
        FileNotFoundError: if the cache doesn't exist (then run the embedding
        comparison first: "python -m app.embeddings --compare")
        """
        cache_path = self.config.embeddings_cache_dir / "text-embedding-3-large.json"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cached embeddings not found at {cache_path}. "
                "Run python -m app.embeddings --compare first."
            )

        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)

        self.corpus_embeddings = cached["embeddings"]
        print(
            f" Loaded {len(self.corpus_embeddings)} cached embeddings (text-embedding-3-large)"
        )

    # Retrieval

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """for the cosine similarity between two vectors, it returns 0.0 for zero vectors"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve_context(self, query_text: str) -> list[dict]:
        """
        this retrieves the top-k most relevant chunks for a query via cosine similarity

        It embeds the query with "text-embedding-3-large" (one OpenAI API call),
        and then compares against all cached corpus embeddings in memory

        Returns:
        self.TOP_K chunk dicts (keys: text, source_id, article, etc.)
        """
        # lazy import
        from app.embeddings.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-3-large", dimensions=3072)
        query_emb, _, _ = embedder.embed_query(query_text)

        similarities = [
            (i, self._cosine_similarity(query_emb, emb))
            for i, emb in enumerate(self.corpus_embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[: self.TOP_K]]
        return [self.chunks[i] for i in top_indices]

    # Prompt formatting

    @staticmethod
    def _format_chunk(chunk: dict, index: int) -> str:
        """
        this formats a chunk as a numbered context passage for the LLM prompt

        chunks longer > 800 characters are truncated to keep prompt size manageable
        """
        source = chunk.get("source_id", "unknown").upper()
        source_map = {
            "GDPR": "GDPR",
            "FADP": "Swiss FADP",
            "EDPB_CONSENT": "EDPB Guidelines on Consent",
            "EDPB_ARTICLE48": "EDPB Guidelines on Article 48 Transfers",
            "EDPB_LEGITIMATE_INTEREST": "EDPB Guidelines on Legitimate Interest",
            "FDPIC_TECHNICAL_MEASURES": "FDPIC Guide to Technical Measures",
        }
        source_label = source_map.get(source, source)

        location_parts = []
        if chunk.get("article"):
            location_parts.append(chunk["article"])
        if chunk.get("section"):
            location_parts.append(chunk["section"])
        if chunk.get("paragraph"):
            location_parts.append(chunk["paragraph"])
        location = ", ".join(location_parts) if location_parts else "General"

        text = chunk.get("text", "").strip()
        if len(text) > 800:
            text = text[:800] + "..."

        return f'[{index}] Source: {source_label}, {location}\n"{text}"'

    def format_user_prompt(self, query_text: str, context_chunks: list[dict]) -> str:
        """
        this builds user prompt: numbered context passages followed by the question,
        and separated by '---'
        """
        context_parts = [
            self._format_chunk(c, i + 1) for i, c in enumerate(context_chunks)
        ]
        context_block = "\n\n".join(context_parts)
        return (
            f"Context passages:\n\n{context_block}\n\n"
            f"---\n\n"
            f"Question: {query_text}"
        )

    # Response evaluation

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """
        This extracts and normalise article references from the response text

        It matches "Article 17", "Art. 25", "Art.6", "Recital 26",
        all normalised to "Article N"
        Returns a sorted and deduplicated list
        """
        patterns = [
            r"Article\s+(\d+)",
            r"Art\.\s*(\d+)",
            r"Recital\s+(\d+)",
        ]
        found = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                found.add(f"Article {match.group(1)}")
        return sorted(found)

    @staticmethod
    def detect_uncertainty(text: str) -> bool:
        """
        this returns true if the response contains any phrase from "UNCERTAINTY_PHRASES"
        (e.g., "insufficient evidence", "cannot determine", "not addressed")
        """
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in UNCERTAINTY_PHRASES)

    def evaluate_response(
        self,
        response: LLMResponse,
        query: BenchmarkQuery,
        context_chunks: list[dict],
    ) -> LLMEvaluationResult:
        """
        This calculates citation precision/recall and calibration for one response

        citation precision = correct_found / total_found (well-evidenced queries)
        citation recall    = correct_found / total_expected

        calibration: correct when confidence matches evidence availability,
        well-evidenced : should answer, under-evidenced : should refuse
        """
        citations_found = self.extract_citations(response.response_text)
        expressed_uncertainty = self.detect_uncertainty(response.response_text)

        # citation precision
        if citations_found and query.expected_citations:
            expected_set = {c for c in query.expected_citations}
            correct = sum(1 for c in citations_found if c in expected_set)
            precision = correct / len(citations_found) if citations_found else 0.0
        elif not citations_found:
            # no citations found
            precision = 1.0 if not query.expected_citations else 0.0
        else:
            # citations found but none expected (under-evidenced query)
            precision = 0.0

        # citation recall
        if query.expected_citations:
            found_set = set(citations_found)
            recalled = sum(1 for c in query.expected_citations if c in found_set)
            recall = recalled / len(query.expected_citations)
        else:
            recall = 1.0  # nothing to miss

        # calibration
        if query.evidence_sufficient:
            calibration_correct = not expressed_uncertainty
        else:
            calibration_correct = expressed_uncertainty

        return LLMEvaluationResult(
            model_label=response.model_label,
            query_id=response.query_id,
            query_category=response.query_category,
            evidence_sufficient=query.evidence_sufficient,
            citations_found=citations_found,
            citations_expected=query.expected_citations,
            citation_precision=round(precision, 3),
            citation_recall=round(recall, 3),
            expressed_uncertainty=expressed_uncertainty,
            calibration_correct=calibration_correct,
            latency_ms=round(response.latency_ms, 1),
            cost_usd=response.cost_usd,
            response_text=response.response_text,
        )

    # main loop for comparison

    def run(
        self, models: list[LLMModelConfig] | None = None
    ) -> list[LLMEvaluationResult]:
        """
        this runs the full comparison across all models and benchmark queries

        The context is retrieved once per query and shared across all the models to
        ensure a fair comparison

        Returns 2 x 8 = 16 LLMEvaluationResult objects

        Args:
        models: the model configs to compare, it defaults to LLM_MODELS from config

        Raises:
        RuntimeError: if the TOGETHER_API_KEY is not set
        """
        if not self.chunks:
            self.load_corpus()
        if self.corpus_embeddings is None:
            self.load_embeddings()

        self.config.ensure_dirs()

        if not self.client.is_available():
            raise RuntimeError("TOGETHER_API_KEY is not set")

        models = models or LLM_MODELS
        queries = BENCHMARK_QUERIES
        all_results: list[LLMEvaluationResult] = []

        # the pre-retrieve context for all the queries so that every model gets identical evidence
        print("\n Retrieving context for benchmark queries...")
        query_contexts: dict[str, list[dict]] = {}
        for query in queries:
            context = self.retrieve_context(query.text)
            query_contexts[query.query_id] = context
            sources = [
                f"{c.get('source_id', '?')}:{c.get('article') or c.get('paragraph') or '?'}"
                for c in context
            ]
            print(f"    {query.query_id}: {', '.join(sources)}")

        for model in models:
            print(f"\n   {model.label} ")

            for query in queries:
                context_chunks = query_contexts[query.query_id]
                user_prompt = self.format_user_prompt(query.text, context_chunks)

                text, latency_ms, in_tokens, out_tokens = self.client.generate(
                    model=model.name,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=1024,
                    temperature=0.0,  # deterministic temp zero
                )

                cost = (
                    in_tokens * model.cost_per_million_input / 1_000_000
                    + out_tokens * model.cost_per_million_output / 1_000_000
                )

                response = LLMResponse(
                    model_label=model.label,
                    query_id=query.query_id,
                    query_category=query.category,
                    response_text=text,
                    latency_ms=latency_ms,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    cost_usd=cost,
                )

                result = self.evaluate_response(response, query, context_chunks)
                all_results.append(result)

                cal_marker = "OK" if result.calibration_correct else "FAIL"
                print(
                    f"    {query.query_id}: "
                    f"prec={result.citation_precision:.2f}  "
                    f"rec={result.citation_recall:.2f}  "
                    f"unc={'Y' if result.expressed_uncertainty else 'N'}  "
                    f"cal={cal_marker}  "
                    f"({latency_ms:.0f}ms)"
                )

        return all_results

    # Aggregation

    @staticmethod
    def aggregate(results: list[LLMEvaluationResult]) -> list[LLMComparisonResult]:
        """
        this groups results per query by model and computes the summary metrics

        citation metrics are averaged over well-evidenced queries only (under-
        evidenced queries have no expected citations)

        calibration is averaged over all the queries
        Returns one LLMComparisonResult per model
        """
        groups: dict[str, list[LLMEvaluationResult]] = defaultdict(list)
        for r in results:
            groups[r.model_label].append(r)

        avg = lambda vals: sum(vals) / len(vals) if vals else 0.0

        aggregated: list[LLMComparisonResult] = []

        for label, items in sorted(groups.items()):
            well = [r for r in items if r.evidence_sufficient]
            under = [r for r in items if not r.evidence_sufficient]

            cite_prec = avg([r.citation_precision for r in well])
            cite_rec = avg([r.citation_recall for r in well])
            cal_acc = avg([1.0 if r.calibration_correct else 0.0 for r in items])
            ie_rate = (
                avg([1.0 if r.expressed_uncertainty else 0.0 for r in under])
                if under
                else 0.0
            )
            fr_rate = (
                avg([1.0 if r.expressed_uncertainty else 0.0 for r in well])
                if well
                else 0.0
            )

            aggregated.append(
                LLMComparisonResult(
                    model_label=label,
                    avg_citation_precision=round(cite_prec, 3),
                    avg_citation_recall=round(cite_rec, 3),
                    calibration_accuracy=round(cal_acc, 3),
                    insufficient_evidence_rate=round(ie_rate, 3),
                    false_refusal_rate=round(fr_rate, 3),
                    avg_latency_ms=round(avg([r.latency_ms for r in items]), 1),
                    avg_cost_per_query=round(avg([r.cost_usd for r in items]), 6),
                    individual_results=items,
                )
            )

        return aggregated

    # Output

    def save_results(
        self,
        aggregated: list[LLMComparisonResult],
    ) -> Path:
        """
        this saves the comparison results to data/output/llm/comparison_results.json

        JSON structure:
        "aggregated"  : per model summary metrics
        "individual"  : per model and per query detailed results with the response text

        Returns:
        the Path to the saved file
        """
        output = {
            "aggregated": [
                {
                    "model_label": a.model_label,
                    "avg_citation_precision": a.avg_citation_precision,
                    "avg_citation_recall": a.avg_citation_recall,
                    "calibration_accuracy": a.calibration_accuracy,
                    "insufficient_evidence_rate": a.insufficient_evidence_rate,
                    "false_refusal_rate": a.false_refusal_rate,
                    "avg_latency_ms": a.avg_latency_ms,
                    "avg_cost_per_query": a.avg_cost_per_query,
                }
                for a in aggregated
            ],
            "individual": [
                {
                    "model_label": r.model_label,
                    "query_id": r.query_id,
                    "query_category": r.query_category,
                    "evidence_sufficient": r.evidence_sufficient,
                    "citations_found": r.citations_found,
                    "citations_expected": r.citations_expected,
                    "citation_precision": r.citation_precision,
                    "citation_recall": r.citation_recall,
                    "expressed_uncertainty": r.expressed_uncertainty,
                    "calibration_correct": r.calibration_correct,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "response_text": r.response_text,
                }
                for a in aggregated
                for r in a.individual_results
            ],
        }

        path = self.config.results_path()
        # ensure_ascii=False preserves special chars in regulatory text
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return path

    @staticmethod
    def format_results(aggregated: list[LLMComparisonResult]) -> str:
        """
        this formats the aggregated results as a table for the terminal output

        Columns: Model, Cite Prec, Cite Rec, Cal Acc, IE Rate, FR Rate, Lat(ms), $/query
        """
        header = (
            f"{'Model':<20} {'Cite Prec':>10} {'Cite Rec':>10} "
            f"{'Cal Acc':>8} {'IE Rate':>8} {'FR Rate':>8} "
            f"{'Lat(ms)':>8} {'$/query':>10}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for a in aggregated:
            lines.append(
                f"{a.model_label:<20} "
                f"{a.avg_citation_precision:>10.3f} {a.avg_citation_recall:>10.3f} "
                f"{a.calibration_accuracy:>8.3f} {a.insufficient_evidence_rate:>8.3f} "
                f"{a.false_refusal_rate:>8.3f} "
                f"{a.avg_latency_ms:>8.1f} ${a.avg_cost_per_query:>9.6f}"
            )

        lines.append(sep)
        lines.append("")
        lines.append("Cite Prec = citation precision (well-evidenced only)")
        lines.append("Cite Rec  = citation recall (well-evidenced only)")
        lines.append("Cal Acc   = calibration accuracy (all queries)")
        lines.append("IE Rate   = insufficient-evidence rate (under-evidenced only)")
        lines.append("FR Rate   = false refusal rate (well-evidenced only)")

        return "\n".join(lines)
