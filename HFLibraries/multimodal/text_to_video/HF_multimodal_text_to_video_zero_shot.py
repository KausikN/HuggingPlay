"""
HuggingFace - Multimodal - Text to Video - Zero-Shot Text to Video Pipeline

Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py
"""

# Imports
from Utils.Utils import *

import imageio

import torch
from diffusers import TextToVideoZeroPipeline
# from diffusers.utils import export_to_video

# Main Functions
## UI Funcs
def UI_Func_LoadInputs(**params):
    '''
    UI - Load Inputs
    '''
    # Init
    USERINPUT_Inputs = {}
    # Ask Inputs
    ## Prompt
    USERINPUT_Inputs["prompt"] = st.text_input("Enter Prompt", value="Moon revolving around the earth.").strip()
    if USERINPUT_Inputs["prompt"] == "":
        st.error("Invalid Prompt.")
        st.stop()
    ## Height and Width
    cols = st.columns(2)
    USERINPUT_Inputs["height"] = cols[0].number_input("Enter Height", min_value=8, max_value=1024, value=256, step=8)
    USERINPUT_Inputs["width"] = cols[1].number_input("Enter Width", min_value=8, max_value=1024, value=256, step=8)
    ## N Inference Steps and N Frames
    cols = st.columns(2)
    USERINPUT_Inputs["num_inference_steps"] = cols[0].number_input("N Inference Steps", min_value=1, value=50, step=1)
    USERINPUT_Inputs["num_frames"] = cols[1].number_input("N Frames", min_value=1, value=8, step=1)

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS, **params):
    '''
    UI - Display Outputs
    '''
    # Init
    VIDEO_FRAMES = OUTPUTS["video_frames"]
    VIDEO_SAVE_PATH = os.path.join(UTILS_PATHS["temp"], "HF_multimodal_text_to_video.mp4")
    # Save Outputs
    imageio.mimsave(
        VIDEO_SAVE_PATH, VIDEO_FRAMES, 
        fps=16
    )
    # Display Outputs
    st.video(VIDEO_SAVE_PATH)

## HF Funcs
def HF_Func_LoadModel(model_info, **params):
    '''
    HF - Load Model
    '''
    # Init
    HF_ID = model_info["hf_id"]
    HF_OUTPUTPARAMS = {
        "output_key": "images"
    } if "output_params" not in model_info["data"] else model_info["data"]["output_params"]
    MODEL_DATA = {
        "hf_id": HF_ID,
        "hf_params": {
            "cache_dir": HF_CACHE_PATHS["default"]
        },
        "params": {
            "model": {
                "torch_dtype": torch.float16,
                "variant": "memory efficient attention"
            }
        },
        "hf_output_params": HF_OUTPUTPARAMS,
        "pipeline": None
    }
    # Load Params
    MODEL_DATA = safe_update_model_data_dict(MODEL_DATA, model_info)
    # Load Model
    PIPELINE = TextToVideoZeroPipeline.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"],
        **MODEL_DATA["params"]["model"]
    )
    PIPELINE.enable_model_cpu_offload()
    MODEL_DATA["pipeline"] = PIPELINE
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs, **params):
    '''
    HF - Run Model
    '''
    # Init
    PIPELINE = MODEL_DATA["pipeline"]
    OUTPUT_PARAMS = MODEL_DATA["hf_output_params"]
    # Fix Input Keys
    inputs["video_length"] = inputs.pop("num_frames")
    # Run Model
    OUTPUTS_DATA = PIPELINE(**inputs)
    FRAMES = OUTPUTS_DATA.__dict__[OUTPUT_PARAMS["output_key"]]
    FRAMES = [(f * 255).astype("uint8") for f in FRAMES]
    OUTPUTS = {
        "video_frames": FRAMES,
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