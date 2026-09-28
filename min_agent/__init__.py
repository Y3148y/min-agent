"""min_agent -- a minimal viable agent built from scratch.

Design is intentionally small: a ReAct-style loop, a registry of JSON-Schema
declared tools, per-session transcripts, context compaction and a three-tier
memory. No agent framework is involved; the only heavy dependency is the
Anthropic client against whatever compatible endpoint ``ANTHROPIC_BASE_URL``
points at.
"""

__version__ = "0.1.0"