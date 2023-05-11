"""
HuggingFace - Agent - OpenAssistant

Ref: https://huggingface.co/docs/transformers/transformers_agents
"""

# Imports
from Utils.Utils import *

import torch
from huggingface_hub import login
from transformers import HfAgent

from . import HF_agent_starcoder

## Import Vars
URL_ENDPOINT = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

# Main Functions
## Utils Functions

## UI Funcs
def UI_Func_LoadInputs(**params):
    '''
    UI - Load Inputs
    '''
    return HF_agent_starcoder.UI_FUNCS["load_inputs"](**params)

def UI_Func_DisplayOutputs(OUTPUTS, **params):
    '''
    UI - Display Outputs
    '''
    return HF_agent_starcoder.UI_FUNCS["display_outputs"](OUTPUTS, **params)

## HF Funcs
def HF_Func_LoadAgent(model_info, **params):
    '''
    HF - Load Agent
    '''
    # Init
    HF_TOKEN = model_info["token"]
    MODEL_DATA = {
        "token": HF_TOKEN,
        "hf_params": {
            "cache_dir": HF_CACHE_PATHS["default"]
        },
        "params": {
            "agent": {}
        },
        "agent": None
    }
    # Load Params
    MODEL_DATA = safe_update_model_data_dict(MODEL_DATA, model_info)
    # Login to HuggingFace Hub
    if HF_TOKEN is not None: login(HF_TOKEN)
    # Load Agent
    MODEL_DATA["agent"] = HfAgent(
        URL_ENDPOINT, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"],
        **MODEL_DATA["params"]["agent"]
    )
    
    return MODEL_DATA

def HF_Func_RunAgent(MODEL_DATA, inputs, **params):
    '''
    HF - Run Agent
    '''
    return HF_agent_starcoder.HF_FUNCS["run_agent"](MODEL_DATA, inputs, **params)

def HF_Func_ChatAgent_Run(MODEL_DATA, inputs, **params):
    '''
    HF - Chat Agent - Run
    '''
    return HF_agent_starcoder.HF_FUNCS["chat_agent_run"](MODEL_DATA, inputs, **params)

def HF_Func_ChatAgent_Reset(MODEL_DATA, **params):
    '''
    HF - Chat Agent - Reset
    '''
    return HF_agent_starcoder.HF_FUNCS["chat_agent_reset"](MODEL_DATA, **params)

# Main Vars
HF_FUNCS = {
    "load_agent": HF_Func_LoadAgent,
    "run_agent": HF_Func_RunAgent,
    "chat_agent_run": HF_Func_ChatAgent_Run,
    "chat_agent_reset": HF_Func_ChatAgent_Reset
}
UI_FUNCS = {
    "load_inputs": UI_Func_LoadInputs,
    "display_outputs": UI_Func_DisplayOutputs
}