"""
EASE (EXL3 Adaptive Speculative Engine) Package
Production-grade Speculative Decoding Engine for Qwen3.8-27B EXL3 on NVIDIA RTX 3060 12GB.
"""

from .restricted_lm_head import RestrictedLMHeadEXL3
from .fast_lm_head import FastDrafterLMHead
from .ngram_table import CommittedNGramTable
from .candidate_tree import CandidateTree, TreeNode
from .engine import EASEEngine

__all__ = [
    "EASEEngine",
    "FastDrafterLMHead",
    "RestrictedLMHeadEXL3",
    "CommittedNGramTable",
    "CandidateTree",
    "TreeNode",
]
