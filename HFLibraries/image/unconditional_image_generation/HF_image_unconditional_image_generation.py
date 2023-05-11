"""
HuggingFace - Image - Unconditional Image Generation - Standard Pipeline (Diffusion)

Ref: 
"""

# Imports
from Utils.Utils import *

from PIL import Image

import torch
from diffusers import DiffusionPipeline

# Main Functions
## Utils Functions
def Utils_ResizeImage(I, max_size=256):
    '''
    Utils - Resize Image to fit max size in width or height
    '''
    # Init
    NEW_SIZE = [max_size, max_size]
    # Find New Size
    if I.shape[1] > I.shape[0]:
        NEW_SIZE[0] = int((I.shape[0]/I.shape[1]) * max_size)
    elif I.shape[0] > I.shape[1]:
        NEW_SIZE[1] = int((I.shape[1]/I.shape[0]) * max_size)
    # Resize
    I_resized = cv2.resize(I, tuple(NEW_SIZE[::-1]))

    return I_resized

## UI Funcs
def UI_Func_LoadInputs(**params):
    '''
    UI - Load Inputs
    '''
    # Init
    USERINPUT_Inputs = {}
    # Ask Inputs
    ## Image
    cols = st.columns((1, 4))
    if cols[0].checkbox("Image"):
        USERINPUT_Inputs["image"] = cols[1].file_uploader("Input Image", type=["png", "jpg", "jpeg"])
        if USERINPUT_Inputs["image"] is not None:
            ## Process Image
            ImageData = USERINPUT_Inputs["image"].read()
            ImageData = cv2.imdecode(np.frombuffer(ImageData, np.uint8), cv2.IMREAD_COLOR)
            USERINPUT_Inputs["image"] = cv2.cvtColor(ImageData, cv2.COLOR_BGR2RGB)
            ## Display Input Image
            st.image(USERINPUT_Inputs["image"], caption="Input Image", use_column_width=True)
            ## Convert to PIL Image
            USERINPUT_Inputs["image"] = Image.fromarray(USERINPUT_Inputs["image"])
    ## Prompt
    cols = st.columns((1, 4))
    if cols[0].checkbox("Prompt"):
        USERINPUT_Inputs["prompt"] = cols[1].text_input("Prompt", value="A beautiful landscape.").strip()
    ## N Inference Steps and Strength and N Images
    cols = st.columns(3)
    USERINPUT_Inputs["num_inference_steps"] = cols[0].number_input("N Inference Steps", min_value=1, value=50, step=1)
    USERINPUT_Inputs["strength"] = cols[1].number_input("Strength", min_value=0.0, max_value=1.0, value=0.8)
    USERINPUT_Inputs["num_images_per_prompt"] = cols[2].number_input("N Images", min_value=1, value=1, step=1)
    ## Seed
    cols = st.columns((1, 4))
    if cols[0].checkbox("Seed"):
        USERINPUT_Inputs["generator"] = torch.Generator().manual_seed(cols[1].number_input("Seed", value=0, step=1))
    ## Negative Prompt and Guidance Scale
    cols = st.columns((1, 4))
    if cols[0].checkbox("Negative Prompt"):
        USERINPUT_Inputs["negative_prompt"] = cols[1].text_area("Enter Negative Prompt", value="blurry").strip()
        USERINPUT_Inputs["guidance_scale"] = st.number_input("Guidance Scale", min_value=0.0, value=8.0, step=0.1)

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS, **params):
    '''
    UI - Display Outputs
    '''
    # Init
    IMAGES = OUTPUTS["generated_images"]
    IMAGES_SAVE_PATH = os.path.join(UTILS_PATHS["temp"], "HF_image_unconditional_image_generation_{}.png")
    # Save Outputs
    for i in range(len(IMAGES)):
        cv2.imwrite(IMAGES_SAVE_PATH.format(i+1), cv2.cvtColor(IMAGES[i], cv2.COLOR_RGB2BGR))
    # Display Outputs
    for i in range(len(IMAGES)):
        st.image(IMAGES[i], caption=f"Generated Image {i+1}", use_column_width=True)

## HF Funcs
def HF_Func_LoadModel(model_info, **params):
    '''
    HF - Load Model
    '''
    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Init
    HF_ID = model_info["hf_id"]
    MODEL_DATA = {
        "hf_id": HF_ID,
        "hf_params": {
            "cache_dir": HF_CACHE_PATHS["default"]
        },
        "params": {
            "model": {
                "variant": "memory efficient attention"
            }
        },
        "pipeline": None
    }
    # Load Params
    MODEL_DATA = safe_update_model_data_dict(MODEL_DATA, model_info)
    # Load Model
    MODEL_DATA["pipeline"] = DiffusionPipeline.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"],
        **MODEL_DATA["params"]["model"]
    )
    ## To Device
    MODEL_DATA["pipeline"] = MODEL_DATA["pipeline"].to(DEVICE)
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs, **params):
    '''
    HF - Run Model
    '''
    # Init
    PIPELINE = MODEL_DATA["pipeline"]
    # Run Model
    OUTPUT_DATA = PIPELINE(**inputs)
    # Extract Images
    Is = []
    try: Is = OUTPUT_DATA.images
    except:
        try: Is = OUTPUT_DATA["sample"]
        except: pass
    Is = [np.array(I) for I in Is]
    # Form Outputs
    OUTPUTS = {
        "generated_images": Is
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