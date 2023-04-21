"""
Utils
"""

# Env Vars
import os
## Transformers Vars
HF_CACHE_PATH = "Data/_hf/cache/"
os.makedirs(HF_CACHE_PATH, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_PATH

# Imports
import json
import pickle
import functools
import numpy as np
import streamlit as st

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer

# Transformers Info
from transformers import file_utils
print("HF Cache Path (Default):", file_utils.default_cache_path)
print("HF Cache Path (Set):", HF_CACHE_PATH)

# Utils Functions
def name_to_path(name, lower=True):
    # Convert to Lowercase
    if lower: name = name.lower()
    # Remove Special Chars
    for c in [" ", "-", ".", "<", ">", "/", "\\", ",", ";", ":", "'", '"', "|", "*", "?"]:
        name = name.replace(c, "_")

    return name

# Main Vars
## Path Vars
UTILS_PATHS = {
    "temp": "Data/_temp/"
}
os.makedirs(UTILS_PATHS["temp"], exist_ok=True)

# Main Functions
