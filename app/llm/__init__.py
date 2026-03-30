# __init__.py: this is the Public API for the LLM comparison package (app/llm/)
#
# This defines the public API for the package so that other modules can import
# from "app.llm" (e.g., "from app.llm import LLMConfig") instead of from individual
# modules
#
# Package (app/llm/) contains:
# - config.py           : the model IDs, benchmark queries, pricing, and settings
# - models.py           : the data classes for the comparison results
# - together_client.py  : HTTP client for Together AI's API
# - comparison.py       : the comparison harness which calculates citation precision,
#                         recall, calibration, and cost metrics
# - __main__.py         : this is the CLI entry point ("python -m app.llm --compare")
#

"""LLM comparison pipeline"""

# LLMConfig: the model identifiers, eight benchmark queries (5 well evidenced and
# 3 "underevidenced"), the pricing data, and params (e.g., temperature, max tokens)
from app.llm.config import LLMConfig

# LLMComparisonResult: a result for a single query: answer text, citation
# precision/recall, evidence sufficiency flag, latency, and token counts
#
# LLMEvaluationResult: the results aggregated per mopdel: average precision/recall,
# calibration accuracy, insufficient evidence detection / false refusal
# rates, and the cost
from app.llm.models import LLMComparisonResult, LLMEvaluationResult

# this defines the public API for "from app.llm import *"
__all__ = ["LLMConfig", "LLMEvaluationResult", "LLMComparisonResult"]
