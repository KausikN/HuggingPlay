"""
HuggingFace - Image - Depth Estimation - Standard Pipeline (DPT)

Ref: 
"""

# Imports
from Utils.Utils import *

from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

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

def Utils_Display3DImage(I, invert_z=True, preserve_aspect_ratio=True, plot_mode="scatter"):
    '''
    Utils - Display 3D Image
    '''
    # Aspect Ratio
    LIMS = [1.0, 1.0]
    if preserve_aspect_ratio:
        if I.shape[1] > I.shape[0]:
            LIMS[1] = I.shape[0]/I.shape[1]
        elif I.shape[0] > I.shape[1]:
            LIMS[0] = I.shape[1]/I.shape[0]
    # Init
    X, Y = np.linspace(0, LIMS[0], I.shape[1]), np.linspace(0, LIMS[1], I.shape[0])
    Z = np.array(I[:, :, -1], dtype=float) / 255.0
    if invert_z: Z = 1.0 - Z
    C = np.array(I[:, :, :3], dtype=float) / 255.0
    # Choose Plot Type
    if plot_mode.lower() == "surface":
        # Plot - Surface Plot
        facecolors = np.array(C * 255.0, dtype=np.uint8)
        I_8bit = Image.fromarray(facecolors).convert("P", palette="WEB", dither=None)
        I_idx = Image.fromarray(facecolors).convert("P", palette="WEB")
        idx_to_color = np.array(I_idx.getpalette()).reshape((-1, 3))
        colorscale = [[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
        FIG = go.Figure(data=[go.Surface(
            z=Z, surfacecolor=I_8bit, cmin=0, cmax=255, colorscale=colorscale, showscale=False
        )])
    else:
        # Plot - Scatter Plot
        xx, yy = np.meshgrid(X, Y)
        xx, yy, zz = xx.flatten(), yy.flatten(), Z.flatten()
        cc = C.reshape((-1, 3))
        FIG = go.Figure(
            data=[go.Scatter3d(
                x=xx, y=yy, z=zz, 
                mode="markers",
                marker={
                    "color": [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in cc],
                    "size": 3
                }
            )]
        )

    # Update Layout
    FIG.update_layout(
        title="Depth", autosize=False,
        width=500, height=500,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    return FIG

## UI Funcs
def UI_Func_LoadInputs(**params):
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
    ## Image
    USERINPUT_Inputs["processor"]["image"] = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])
    if USERINPUT_Inputs["processor"]["image"] is None:
        st.error("No image provided.")
        st.stop()
    # Process Inputs
    ImageData = USERINPUT_Inputs["processor"]["image"].read()
    ImageData = cv2.imdecode(np.frombuffer(ImageData, np.uint8), cv2.IMREAD_COLOR)
    USERINPUT_Inputs["processor"]["image"] = cv2.cvtColor(ImageData, cv2.COLOR_BGR2RGB)
    ## Display Input Image
    st.image(USERINPUT_Inputs["processor"]["image"], caption="Input Image", use_column_width=True)

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(
        OUTPUTS, 
        interactive_display=True, 
        invert_z=True, preserve_aspect_ratio=True, plot_mode="scatter",
        **params
    ):
    '''
    UI - Display Outputs
    '''
    # Init
    PLOT_FUNC = st.plotly_chart if interactive_display else st.pyplot
    DEPTH_IMAGE = OUTPUTS["depth_image"]
    IMAGE_SAVE_PATH = os.path.join(UTILS_PATHS["temp"], "HF_image_depth_estimation.png")
    # Save Outputs
    DEPTH_IMAGE = Image.fromarray(DEPTH_IMAGE)
    DEPTH_IMAGE.save(IMAGE_SAVE_PATH)
    # Display Outputs
    ## Side-by-Side
    cols = st.columns(2)
    cols[0].image(OUTPUTS["image"], caption="Input Image", use_column_width=True)
    cols[1].image(IMAGE_SAVE_PATH, caption="Depth Image", use_column_width=True)
    ## 3D Plot
    FULL_IMAGE = np.concatenate([OUTPUTS["image"], np.expand_dims(DEPTH_IMAGE, axis=-1)], axis=-1)
    # plt.imshow(FULL_IMAGE)
    FIG = Utils_Display3DImage(
        FULL_IMAGE, 
        invert_z=invert_z, preserve_aspect_ratio=preserve_aspect_ratio,
        plot_mode=plot_mode
    )
    PLOT_FUNC(FIG)

## HF Funcs
def HF_Func_LoadModel(model_info, **params):
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
    MODEL_DATA["processor"] = DPTImageProcessor.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"]
    )
    MODEL_DATA["model"] = DPTForDepthEstimation.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"]
    )
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs, **params):
    '''
    HF - Run Model
    '''
    # Init
    MODEL = MODEL_DATA["model"]
    # Run Model
    PROC_INPUTS = MODEL_DATA["processor"](
        images=inputs["processor"]["image"], 
        **MODEL_DATA["params"]["processor"]
    )
    with torch.no_grad():
        OUTPUT_DATA = MODEL(**PROC_INPUTS)
        OUTPUT_DATA = OUTPUT_DATA.predicted_depth
    # Interpolate to original size
    OUTPUT_DATA = torch.nn.functional.interpolate(
        OUTPUT_DATA.unsqueeze(1),
        size=inputs["processor"]["image"].shape[:2],
        mode="bicubic",
        align_corners=False
    )
    # Convert to numpy array
    OUTPUT_DATA = OUTPUT_DATA.squeeze().cpu().numpy()
    OUTPUT_DATA = (OUTPUT_DATA * 255 / np.max(OUTPUT_DATA)).astype("uint8")
    # Form Outputs
    OUTPUTS = {
        "image": inputs["processor"]["image"],
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