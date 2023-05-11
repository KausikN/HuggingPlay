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
    "current": "Data/_hf/cache/"
}
## Transformers Functions
def set_hf_cache_path(path=HF_CACHE_PATHS["default"]):
    '''
    Utils - Set HF Cache Path
    '''
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
    '''
    Utils - Name to Path
    '''
    # Convert to Lowercase
    if lower: name = name.lower()
    # Remove Special Chars
    for c in [" ", "-", ".", "<", ">", "/", "\\", ",", ";", ":", "'", '"', "|", "*", "?"]:
        name = name.replace(c, "_")

    return name

def safe_update_model_data_dict(MODEL_DATA, model_info):
    '''
    Utils - Safe Update Model Data Dict
    '''
    # Update HF Params (2 levels)
    for k in MODEL_DATA["params"].keys():
        if k in model_info["data"]["params"].keys():
            for pk in model_info["data"]["params"][k].keys():
                MODEL_DATA["params"][k][pk] = model_info["data"]["params"][k][pk]
    # Update HF Cache Params (1 level)
    for k in MODEL_DATA["hf_params"].keys():
        if k in model_info["data"]["hf_params"].keys():
            MODEL_DATA["hf_params"][k] = model_info["data"]["hf_params"][k]

    return MODEL_DATA

# Main Vars
## Path Vars
UTILS_PATHS = {
    "temp": "Data/_temp/"
}
os.makedirs(UTILS_PATHS["temp"], exist_ok=True)

# Main Functions
