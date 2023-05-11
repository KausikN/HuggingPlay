"""
HuggingFace - Image - Depth Estimation - GLPN Pipeline

Ref: 
"""

# Imports
from Utils.Utils import *

from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

import torch
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation

from . import HF_image_depth_estimation

# Main Functions
## UI Funcs
def UI_Func_LoadInputs(**params):
    '''
    UI - Load Inputs
    '''
    return HF_image_depth_estimation.UI_FUNCS["load_inputs"](**params)

def UI_Func_DisplayOutputs(OUTPUTS, interactive_display=True, invert_z=True, preserve_aspect_ratio=True, **params):
    '''
    UI - Display Outputs
    '''
    return HF_image_depth_estimation.UI_FUNCS["display_outputs"](
        OUTPUTS, 
        interactive_display=interactive_display, 
        invert_z=invert_z, preserve_aspect_ratio=preserve_aspect_ratio,
        **params
    )

## HF Funcs
def HF_Func_LoadModel(model_info):
    '''
    HF - Load Model
    '''
    # Init
    HF_ID = model_info["hf_id"]
    MODEL_DATA = {
        "hf_id": HF_ID,
        "hf_params": {
            "cache_dir": HF_CACHE_PATHS["default"]
        },
        "params": {
            "processor": {
                "return_tensors": "pt"
            },
            "model": {}
        },
        "processor": None,
        "model": None
    }
    # Load Params
    MODEL_DATA = safe_update_model_data_dict(MODEL_DATA, model_info)
    # Load Model
    MODEL_DATA["processor"] = GLPNFeatureExtractor.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"]
    )
    MODEL_DATA["model"] = GLPNForDepthEstimation.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"]
    )
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs, **params):
    '''
    HF - Run Model
    '''
    return HF_image_depth_estimation.HF_FUNCS["run_model"](MODEL_DATA, inputs)

# Main Vars
HF_FUNCS = {
    "load_model": HF_Func_LoadModel,
    "run_model": HF_Func_RunModel
}
UI_FUNCS = {
    "load_inputs": UI_Func_LoadInputs,
    "display_outputs": UI_Func_DisplayOutputs
}