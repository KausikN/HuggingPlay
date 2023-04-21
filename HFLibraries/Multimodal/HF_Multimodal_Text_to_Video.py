"""
HuggingFace - Multimodal - Text to Video

Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
"""

# Imports
from Utils.Utils import *

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Main Functions
## UI Funcs
def UI_Func_LoadInputs():
    '''
    UI - Load Inputs
    '''
    # Init
    USERINPUT_Inputs = {}
    # Ask Inputs
    ## Prompt
    USERINPUT_Inputs["prompt"] = st.text_input("Enter Prompt").strip()
    if USERINPUT_Inputs["prompt"] == "":
        st.error("Invalid Prompt.")
        st.stop()
    # Height and Width
    cols = st.columns(2)
    USERINPUT_Inputs["height"] = cols[0].number_input("Enter Height", min_value=8, max_value=1024, value=256, step=8)
    USERINPUT_Inputs["width"] = cols[1].number_input("Enter Width", min_value=8, max_value=1024, value=256, step=8)
    ## N Inference Steps
    USERINPUT_Inputs["num_inference_steps"] = st.number_input("N Inference Steps", min_value=1, value=50, step=1)
    ## 

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS):
    '''
    UI - Display Outputs
    '''
    # Init
    VIDEO_FRAMES = OUTPUTS["video_frames"]
    VIDEO_SAVE_PATH = os.path.join(UTILS_PATHS["temp"], "HF_Multimodal_Text_to_Video.mp4")
    # Save Outputs
    export_to_video(VIDEO_FRAMES, VIDEO_SAVE_PATH)
    # Display Outputs
    st.video(VIDEO_SAVE_PATH)

## HF Funcs
def HF_Func_LoadModel(model_info):
    '''
    HF - Load Model
    '''
    # Init
    HF_ID = model_info["hf_id"]
    HF_PARAMS = {} if "params" not in model_info["data"] else model_info["data"]["params"]
    MODEL_DATA = {
        "hf_id": HF_ID,
        "hf_data": model_info["data"],
        "pipe": None,
    }
    # Load Model
    PIPE = DiffusionPipeline.from_pretrained(
        HF_ID, 
        torch_dtype=torch.float16,
        **HF_PARAMS
    )
    PIPE.scheduler = DPMSolverMultistepScheduler.from_config(PIPE.scheduler.config)
    PIPE.enable_model_cpu_offload()
    MODEL_DATA["pipe"] = PIPE
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs):
    '''
    HF - Run Model
    '''
    # Init
    PIPE = MODEL_DATA["pipe"]
    OUTPUTS = {
        "video_frames": None
    }
    # Run Model
    
    OUTPUTS["video_frames"] = PIPE(**inputs).frames

    return OUTPUTS

# Main Vars
HF_FUNCS = {
    "load_model": HF_Func_LoadModel,
    "run_model": HF_Func_RunModel
}
UI_FUNCS = {
    "load_inputs": UI_Func_LoadInputs,
    "display_outputs": UI_Func_DisplayOutputs
}