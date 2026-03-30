"""
This is for the Isaacus "kanon-2-embedder" via the direct REST API.

"kanon-2-embedder" is a legal domain specific embedding model by Isaacus, which
has been trained on European legal corpora. It was accessed via REST because
Isaacus had no official Python SDK.

The comparison results are as follows (for the GraphLex AI Layer 3 evaluation):
P@5=0.583 (vs. 0.600 for "text-embedding-3-large"). This embedder was excellent
on negation-sensitive queries (P@5=0.733 vs. 0.600), but it was 5.8x slower and 4.5x
more expensive. For this reason, "text-embedding-3-large" was ultimately
chosen for production usage.

API: POST https://api.isaacus.com/v1/embeddings
Body: { "model": "kanon-2-embedder", "texts": [...], "task": "...", "dimensions": 1792 }
Response: { "embeddings": [{"index": 0, "embedding": [...]}, ...], "usage": {"input_tokens": N} }
Docs: https://docs.isaacus.com/models/introduction#embedding
Auth: ISAACUS_API_KEY env variable
"""

# import libraries
from __future__ import annotations
import os
import time
import requests


class KanonEmbedder:
    """
    This class wraps the Isaacus embeddings API for the "kanon-2-embedder"

    Usage:
    embedder = KanonEmbedder(dimensions=1792)
    if embedder.is_available():
    embedding, tokens, ms = embedder.embed_query("What is GDPR Article 17?")
    """

    # all of the embedding requests are posted to this endpoint
    BASE_URL = "https://api.isaacus.com/v1/embeddings"

    def __init__(self, dimensions: int = 1792) -> None:
        """
        Parameters:
        dimensions : int, default 1792
        This is the output vector size. "kanon-2-embedder" doesn't support
        MRL dimension reduction, therefore, this is always 1,792 as a value.
        """
        self.dimensions = dimensions
        self.model = "kanon-2-embedder"

    def is_available(self) -> bool:
        """
        This returns true if the ISAACUS_API_KEY is set and not empty.
        It allows for the comparison harness to skip kanon if the key is not there.
        """
        return bool(os.getenv("ISAACUS_API_KEY"))

    def _api_key(self) -> str:
        """
        This function retrieves the ISAACUS_API_KEY from the env

        Raises:
        ValueError
        If the ISAACUS_API_KEY is not set or it is empty.
        """
        key = os.getenv("ISAACUS_API_KEY", "")
        if not key:
            raise ValueError("ISAACUS_API_KEY not set")
        return key

    def _call(
        self,
        texts: list[str],
        task: str = "retrieval/document",
    ) -> tuple[list[list[float]], int]:
        """
        This sends the embedding request to the Isaacus API

        Parameters:
        texts: list[str]
        The strings to embed (1 for a query, and up to batch_size for corpus chunks).
        task: str
        "retrieval/document" for the stored corpus chunks,
        "retrieval/query" for the search queries.

        Returns:
        a tuple of (embeddings, input_tokens)

        Raises:
        requests.HTTPError
        e.g., 401 bad key, 429 rate limit, 500 server error.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

        # this builds the request body, it's the JSON payload sent to the API
        body: dict = {
            "model": self.model,
            "texts": texts,
            "task": task,
            "dimensions": self.dimensions,
        }

        # sends the HTTP POST request and waits for the answer
        resp = requests.post(self.BASE_URL, json=body, headers=headers, timeout=120)
        # checks for errors, if status is 200 OK then nothing happens
        resp.raise_for_status()

        # parses the JSON response
        data = resp.json()

        # this sorts the embeddings by index
        items = sorted(data["embeddings"], key=lambda x: x["index"])
        # this extracts just the embedding vectors and discards the metadata
        embeddings = [item["embedding"] for item in items]

        # extracts the token count from the "usage" field
        # .get() with defaults handles possible missing or incomplete usage fields
        tokens = data.get("usage", {}).get("input_tokens", 0)

        return embeddings, tokens

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 50,
    ) -> tuple[list[list[float]], int, float]:
        """
        This function embeds a list of texts in batches, it is used for the document corpus

        Parameters:
        texts: list[str]
        All of the texts to embed.
        batch_size: int, default 50
        The texts per API request.

        Returns:
        a tuple of (embeddings, total_tokens, total_ms)
        """
        # for collecting the embedding vectors from all batches
        all_embeddings: list[list[float]] = []
        # for collection of the total tokens from all batches
        total_tokens = 0
        # this starts the timer
        start = time.perf_counter()

        # for loop that processes the texts in batches
        for i in range(0, len(texts), batch_size):
            # slices out the current batch
            batch = texts[i : i + batch_size]
            # calls the API with task="retrieval/document"
            embs, tokens = self._call(batch, task="retrieval/document")
            # appends the embeddings from this batch
            all_embeddings.extend(embs)
            # adds the token count from this batch to the total
            total_tokens += tokens

        # calculates the total elapsed time in ms
        elapsed_ms = (time.perf_counter() - start) * 1000
        return all_embeddings, total_tokens, elapsed_ms

    def embed_query(self, text: str) -> tuple[list[float], int, float]:
        """
        This function embeds a single query text. It is used for each test query during the comparison.

        It uses task="retrieval/query".

        Parameters:
        text: str
        This is the query to embed.

        Returns:
        a tuple of (embedding, tokens, ms)
        Milliseconds is for the latency per query (for kanon it averaged 1,358.6ms).
        """
        # starts the timer for this individual query embedding
        start = time.perf_counter()
        # calls the API with the query text
        embs, tokens = self._call([text], task="retrieval/query")
        # calculates the elapsed time in ms
        elapsed_ms = (time.perf_counter() - start) * 1000
        # returns the first and only embedding from the response
        return embs[0], tokens, elapsed_ms
