"""
HuggingFace - Image - Depth Estimation - Standard Pipeline (DPT)

Ref: 
"""

# Imports
from Utils.Utils import *

from PIL import Image

import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Main Functions
## UI Funcs
def UI_Func_LoadInputs():
    '''
    UI - Load Inputs
    '''
    # Init
    USERINPUT_Inputs = {
        "processor": {
            "image": None
        }
    }
    # Ask Inputs
    ## Prompt
    USERINPUT_Inputs["processor"]["image"] = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])
    if USERINPUT_Inputs["processor"]["image"] is None:
        st.error("No image provided.")
        st.stop()
    # Process Inputs
    ImageData = USERINPUT_Inputs["processor"]["image"].read()
    ImageData = cv2.imdecode(np.frombuffer(ImageData, np.uint8), cv2.IMREAD_COLOR)
    USERINPUT_Inputs["processor"]["image"] = cv2.cvtColor(ImageData, cv2.COLOR_BGR2RGB)

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS):
    '''
    UI - Display Outputs
    '''
    # Init
    DEPTH_IMAGE = OUTPUTS["depth_image"]
    IMAGE_SAVE_PATH = os.path.join(UTILS_PATHS["temp"], "HF_image_depth_estimation.png")
    # Save Outputs
    DEPTH_IMAGE = Image.fromarray(DEPTH_IMAGE)
    DEPTH_IMAGE.save(IMAGE_SAVE_PATH)
    # Display Outputs
    st.image(IMAGE_SAVE_PATH, caption="Depth Image", use_column_width=True)

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
    MODEL_DATA["processor"] = DPTImageProcessor.from_pretrained(HF_ID)
    MODEL_DATA["model"] = DPTForDepthEstimation.from_pretrained(HF_ID)
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs):
    '''
    HF - Run Model
    '''
    # Init
    MODEL = MODEL_DATA["model"]
    # Run Model
    PROC_INPUTS = MODEL_DATA["processor"](images=inputs["processor"]["image"], **MODEL_DATA["hf_params"]["processor"])
    with torch.no_grad():
        OUTPUT_DATA = MODEL(**PROC_INPUTS)
        OUTPUT_DATA = OUTPUT_DATA.predicted_depth
    # Interpolate to original size
    OUTPUT_DATA = torch.nn.functional.interpolate(
        OUTPUT_DATA.unsqueeze(1),
        size=inputs["processor"]["image"].shape[::-1],
        mode="bicubic",
        align_corners=False
    )
    # Convert to numpy array
    OUTPUT_DATA = OUTPUT_DATA.squeeze().cpu().numpy()
    OUTPUT_DATA = (OUTPUT_DATA * 255 / np.max(OUTPUT_DATA)).astype("uint8")
    # Form Outputs
    OUTPUTS = {
        "depth_image": OUTPUT_DATA
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