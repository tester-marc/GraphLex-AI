"""This script is for the OpenAI "text-embedding-3" embedder.

It wraps OpenAI's Embeddings API for the "text-embedding-3-small/large" family of models.

What are embeddings? Embeddings convert chunks of the regulatory text (Swiss FADP, EU GDPR)
into vectors that are stored in Weaviate for the semantic similarity search.

Four different configurations are tested (see also config.py):
- "text-embedding-3-small" at 1536d, 512d, 256d (reduced with Matryoshka Representation Learning (MRL))
- "text-embedding-3-large" at 3072 dimensions

MRL (Matryoshka Representation Learning) lets the API truncate the vectors to less
dimensions via the "dimensions" parameter.

The winner of this comparison here was "text-embedding-3-large" (3072dim), with:
Precision@5=0.600, MRR=0.892 across the 12 test queries. This embedder was selected
as the one to be used for production.
"""

# import libraries
from __future__ import annotations
import os
import time
from openai import OpenAI


class OpenAIEmbedder:
    """This class wraps the OpenAI embeddings API for "text-embedding-3-small/large"

    - "embed_batch()": to embed many texts at once (the corpus of 817 regulatory chunks)
    - "embed_query()": to embed a single query at the search time

    Both of these return the timing information for the reporting of the latency.

    Usage:
    embedder = OpenAIEmbedder(model="text-embedding-3-large", dimensions=3072)
    if embedder.is_available():
    embedding, tokens, ms = embedder.embed_query("What is GDPR Article 17?")
    """

    def __init__(
        self, model: str = "text-embedding-3-small", dimensions: int = 1536
    ) -> None:
        """
        Args:
        model: the OpenAI model identifier ("text-embedding-3-small" or
        "text-embedding-3-large").
        dimensions: This is the output vector size. Sizes are 1536d (for small) and
        3072d (for large) and the lower values use MRL (Matryoshka Representation Learning) truncation
        """
        self.model = model
        self.dimensions = dimensions
        # initialises on first use via "client" property
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """This function returns the OpenAI client and creates it on the first access.
        It automatically authenticates with the OPENAI_API_KEY env variable.
        """
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def is_available(self) -> bool:
        """This returns true if the OPENAI_API_KEY is set and it is not empty, otherwise it returns false."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> tuple[list[list[float]], int, float]:
        """In order to embed a list of texts in batches, it is used to embed the entire corpus.

        Args:
        texts: the texts to embed (817 regulatory chunks for GraphLex AI)
        batch_size: the texts per API request

        Returns:
        - embeddings: 1 vector per input text, in the order of input
        - total_tokens: the total tokens consumed across all the batches
        - total_ms: the time for all API calls in ms
        """
        all_embeddings: list[list[float]] = [[] for _ in texts]
        total_tokens = 0
        start = time.perf_counter()

        # for loop to process texts in batches
        for i in range(0, len(texts), batch_size):
            # slices out the current batch
            batch = texts[i : i + batch_size]
            # calls the OpenAI Embeddings API for this batch
            resp = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )

            # to place each embedding into the correct position in the output list
            # uses item.index as the API can return items that are out of order
            # adds batch offset "i" in order to get the global index
            for item in resp.data:
                all_embeddings[i + item.index] = item.embedding

            # collects the token count from this batch
            total_tokens += resp.usage.total_tokens

        # how much time has elapsed in ms
        elapsed_ms = (time.perf_counter() - start) * 1000
        # returns the three values
        return all_embeddings, total_tokens, elapsed_ms

    def embed_query(self, text: str) -> tuple[list[float], int, float]:
        """This function embeds a single query text and is used at search time

        Args:
        text: the query string to embed

        Returns:
        - embedding: a single vector of length self.dimensions
        - tokens: the tokens consumed
        - ms: the API call latency in ms
        """
        # starts the timer for this query embedding
        start = time.perf_counter()
        # calls the OpenAI API with single text string
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        # how long this API call took
        elapsed_ms = (time.perf_counter() - start) * 1000
        # returns embedding vector, token count, and latency
        return resp.data[0].embedding, resp.usage.total_tokens, elapsed_ms
