"""
HuggingFace - Agent - OpenAI

Ref: 
"""

# Imports
from Utils.Utils import *

import torch
from transformers import OpenAiAgent

from . import HF_agent_starcoder

## Import Vars

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
    OPENAI_TOKEN = model_info["token"]
    MODEL_DATA = {
        "token": OPENAI_TOKEN,
        "hf_data": model_info["data"],
        "hf_params": {
            "agent": {
                "additional_tools": []
            }
        },
        "agent": None
    }
    # Load Params
    if "params" in model_info["data"].keys():
        for k in MODEL_DATA["hf_params"].keys():
            if k in model_info["data"]["params"].keys():
                for pk in model_info["data"]["params"][k].keys():
                    MODEL_DATA["hf_params"][k][pk] = model_info["data"]["params"][k][pk]
    # Load Agent
    MODEL_DATA["agent"] = OpenAiAgent(
        model="text-davinci-003", api_key=OPENAI_TOKEN,
        **MODEL_DATA["hf_params"]["agent"]
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