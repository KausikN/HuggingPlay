"""
HuggingFace - Agent - StarCoder

Ref: https://huggingface.co/docs/transformers/transformers_agents
"""

# Imports
from Utils.Utils import *

import torch
from huggingface_hub import login
from transformers import HfAgent

## Import Vars
URL_ENDPOINT = "https://api-inference.huggingface.co/models/bigcode/starcoder"

# Main Functions
## Utils Functions

## UI Funcs
def UI_Func_LoadInputs(**params):
    '''
    UI - Load Inputs
    '''
    # Init
    USERINPUT_Inputs = {
        "task": "",
        "return_code": False
    }
    # Ask Inputs
    ## Return Code
    USERINPUT_Inputs["return_code"] = st.checkbox("Only Return Code", value=True)
    ## Task
    USERINPUT_Inputs["task"] = st.text_area("Enter Task", height=300).strip()
    if USERINPUT_Inputs["task"] == "":
        st.error("Task is empty.")
        st.stop()

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS, **params):
    '''
    UI - Display Outputs
    '''
    # Init
    # Save Outputs
    # Display Outputs
    st.write(OUTPUTS)

## HF Funcs
def HF_Func_LoadAgent(model_info, **params):
    '''
    HF - Load Agent
    '''
    # Init
    HF_TOKEN = model_info["token"]
    MODEL_DATA = {
        "token": HF_TOKEN,
        "hf_data": model_info["data"],
        "hf_params": {
            "agent": {}
        },
        "agent": None
    }
    # Load Params
    if "params" in model_info["data"].keys():
        for k in MODEL_DATA["hf_params"].keys():
            if k in model_info["data"]["params"].keys():
                for pk in model_info["data"]["params"][k].keys():
                    MODEL_DATA["hf_params"][k][pk] = model_info["data"]["params"][k][pk]
    # Login to HuggingFace Hub
    if HF_TOKEN is not None: login(HF_TOKEN)
    # Load Agent
    MODEL_DATA["agent"] = HfAgent(
        URL_ENDPOINT, 
        **MODEL_DATA["hf_params"]["agent"]
    )
    
    return MODEL_DATA

def HF_Func_RunAgent(MODEL_DATA, inputs, **params):
    '''
    HF - Run Agent
    '''
    # Init
    AGENT = MODEL_DATA["agent"]
    # Run Agent
    OUTPUTS = AGENT.run(**inputs)
    # Form Outputs
    OUTPUTS = OUTPUTS

    return OUTPUTS

def HF_Func_ChatAgent_Run(MODEL_DATA, inputs, **params):
    '''
    HF - Chat Agent - Run
    '''
    # Init
    AGENT = MODEL_DATA["agent"]
    # Run Chat Agent
    OUTPUTS = AGENT.chat(**inputs)
    # Form Outputs
    OUTPUTS = OUTPUTS

    return OUTPUTS

def HF_Func_ChatAgent_Reset(MODEL_DATA, **params):
    '''
    HF - Chat Agent - Reset
    '''
    # Init
    AGENT = MODEL_DATA["agent"]
    # Reset Chat Agent
    AGENT.prepare_for_new_chat()

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