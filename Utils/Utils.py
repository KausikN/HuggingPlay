"""
Utils
"""

# Env Vars
import os
from transformers import file_utils
## Transformers Vars
HF_CACHE_PATHS = {
    "default": file_utils.default_cache_path,
    "local": "Data/_hf/cache/",
    "current": file_utils.default_cache_path
}
## Transformers Functions
def set_hf_cache_path(path=HF_CACHE_PATHS["default"]):
    global HF_CACHE_PATHS
    ## Set
    HF_CACHE_PATHS["current"] = path
    os.makedirs(HF_CACHE_PATHS["current"], exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_PATHS["current"]
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_PATHS["current"]
    file_utils.default_cache_path = HF_CACHE_PATHS["current"]
    ## Display
    print("HF Cache Path (Current):", file_utils.default_cache_path)

# Imports
import cv2
import json
import pickle
import functools
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer

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
