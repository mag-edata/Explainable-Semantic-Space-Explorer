"""
token_definition.py

This module defines the building blocks of tokens that are valid across
the entire project.

The goal is to fix the definition in a single location, guaranteeing
consistency and reproducibility of preprocessing throughout vocabulary
generation, corpus generation, model training, and inference.

When the token definition needs to change, this file is the only place
that should be modified.
"""

import re

# Pattern for extracting alphabetic sequences (used to generate token candidates)
TOKEN_EXTRACT_PATTERN = re.compile(r"[a-zA-Z]+")
# Filtering constraint for tokens composed only of alphabetic characters
TOKEN_CONSTRAINT_PATTERN = re.compile(r"^[a-zA-Z]+$")
