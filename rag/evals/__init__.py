"""Evaluation suite: ``python -m rag.evals --help``."""

from rag.evals.dataset import EvalExample, load_dataset
from rag.evals.runner import EvalReport, run_eval

__all__ = ["EvalExample", "EvalReport", "load_dataset", "run_eval"]
