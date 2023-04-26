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
        "hf_data": model_info["data"],
        "hf_params": {
            "processor": {
                "return_tensors": "pt"
            },
            
        },
        "processor": None,
        "model": None
    }
    # Load Params
    if "params" in model_info["data"].keys():
        for k in MODEL_DATA["hf_params"].keys():
            if k in model_info["data"]["params"].keys():
                for pk in model_info["data"]["params"][k].keys():
                    MODEL_DATA["hf_params"][k][pk] = model_info["data"]["params"][k][pk]
    # Load Model
    MODEL_DATA["processor"] = GLPNFeatureExtractor.from_pretrained(HF_ID)
    MODEL_DATA["model"] = GLPNForDepthEstimation.from_pretrained(HF_ID)
    
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