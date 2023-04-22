"""
HuggingFace - Multimodal - Text to Video - Zero-Shot Text to Video Pipeline

Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py
"""

# Imports
from Utils.Utils import *

import torch
from diffusers import TextToVideoZeroPipeline
# from diffusers.utils import export_to_video
import imageio

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
    ## N Inference Steps and N Frames
    cols = st.columns(2)
    USERINPUT_Inputs["num_inference_steps"] = cols[0].number_input("N Inference Steps", min_value=1, value=50, step=1)
    USERINPUT_Inputs["video_length"] = cols[1].number_input("N Frames", min_value=1, value=8, step=1)

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS):
    '''
    UI - Display Outputs
    '''
    # Init
    VIDEO_FRAMES = OUTPUTS["video_frames"]
    VIDEO_SAVE_PATH = os.path.join(UTILS_PATHS["temp"], "HF_multimodal_text_to_video.mp4")
    # Save Outputs
    imageio.mimsave(
        VIDEO_SAVE_PATH, 
        [(f * 255).astype("uint8") for f in VIDEO_FRAMES], 
        fps=16
    )
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
    HF_OUTPUTPARAMS = {
        "output_key": "images"
    } if "output_params" not in model_info["data"] else model_info["data"]["output_params"]
    MODEL_DATA = {
        "hf_id": HF_ID,
        "hf_data": model_info["data"],
        "hf_params": HF_PARAMS,
        "hf_output_params": HF_OUTPUTPARAMS,
        "pipe": None
    }
    # Load Model
    PIPE = TextToVideoZeroPipeline.from_pretrained(
        HF_ID, 
        torch_dtype=torch.float16,
        **HF_PARAMS
    )
    PIPE.enable_model_cpu_offload()
    MODEL_DATA["pipe"] = PIPE
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs):
    '''
    HF - Run Model
    '''
    # Init
    PIPE = MODEL_DATA["pipe"]
    OUTPUT_PARAMS = MODEL_DATA["hf_output_params"]
    # Run Model
    OUTPUTS_DATA = PIPE(**inputs)
    OUTPUTS = {
        "video_frames": OUTPUTS_DATA.__dict__[OUTPUT_PARAMS["output_key"]],
        **{k: OUTPUTS_DATA.__dict__[k] for k in OUTPUTS_DATA.__dict__.keys() if k != OUTPUT_PARAMS["output_key"]}
    }

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