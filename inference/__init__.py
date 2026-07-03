"""
inference package
==================
Runtime, model-backed components.

This package holds the **single sanctioned exception** to the project rule
"no model at inference time": the sentence-context mode (requirements v2.0,
CONST-03 as amended) encodes the user's input sentence with the locally
deployed transformer to produce an in-context token vector. No network access
and no training are performed here.

Keeping these components in a dedicated package preserves the purity of
``core/`` (which only compares pre-saved vectors).
"""
