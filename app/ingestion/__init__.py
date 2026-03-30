# This is the public API for the GraphLex AI Ingestion Layer (Layer 1)
#
# It re-exports the key classes so that external code can write
# "from app.ingestion import LegalChunker" instead of importing
# it from the internal modules directly
#
# Responsibilities of this layer:
#
# 1. Extraction: To read in text from the regulatory PDFs (GDPR, Swiss FADP, EDPB
# guidelines, and FDPIC guide) through one of 3 extractors
# 2. Chunking: Split the extracted text into legal structure aware chunks with max.
# 512 tokens and a 64 token overlap) along with metadata (source, jurisdiction,
# article number, cross references).
# The output feeds into the embeddings and ultimately into Layer 3 (the Weaviate vector store)
#

from app.ingestion.base_extractor import BaseExtractor  # defines extractor interface
from app.ingestion.chunker import (
    LegalChunker,
)  # splits ExtractionResults into chunks on the article / section boundaries
from app.ingestion.config import (
    IngestionConfig,
)  # dataclass: file paths, document registry, and chunking params
from app.ingestion.mistral_extractor import (
    MistralDocumentAIExtractor,
)  # the cloud extractor via the Mistral API
from app.ingestion.models import (
    Chunk,
    ExtractionResult,
)  # Chunk: the embeddable text segment + the metadata
from app.ingestion.olmocr_extractor import (
    OlmOCRExtractor,
)  # VLM-based extractor (from Allen AI)
from app.ingestion.pymupdf_extractor import (
    PyMuPDFExtractor,
)  # the baseline extractor as well as the one used in production

# __all__ defines the public API
__all__ = [
    "BaseExtractor",
    "IngestionConfig",
    "PyMuPDFExtractor",
    "OlmOCRExtractor",
    "MistralDocumentAIExtractor",
    "LegalChunker",
    "ExtractionResult",
    "Chunk",
]
