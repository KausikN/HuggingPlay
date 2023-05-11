"""
HuggingFace - Text - Text Classification - Standard Pipeline

Ref: 
"""

# Imports
from Utils.Utils import *

import torch
from transformers import AutoModelForSequenceClassification

## Import Vars
AutoModel = AutoModelForSequenceClassification

# Main Functions
## Utils Functions
def Utils_DisplayClassification(prob_dist, classes):
    '''
    Utils - Display Classification
    '''
    FIG = plt.figure()
    plt.bar(list(classes), list(prob_dist))
    plt.title("Text Classification")

    return FIG

## UI Funcs
def UI_Func_LoadInputs(**params):
    '''
    UI - Load Inputs
    '''
    # Init
    USERINPUT_Inputs = {
        "text": ""
    }
    # Ask Inputs
    ## Prompt
    USERINPUT_Inputs["text"] = st.text_area(
        "Enter Text", value="The movie was amazing.", 
        height=300
    ).strip()
    if USERINPUT_Inputs["text"] == "":
        st.error("Text is empty.")
        st.stop()

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS, interactive_display=True, **params):
    '''
    UI - Display Outputs
    '''
    # Init
    PLOT_FUNC = st.plotly_chart if interactive_display else st.pyplot
    # Save Outputs
    # Display Outputs
    FIG = Utils_DisplayClassification(OUTPUTS["prob_dist"], OUTPUTS["classes"])
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
            "tokenizer": {
                "return_tensors": "pt"
            },
            "model": {}
        },
        "config": None,
        "tokenizer": None,
        "model": None,
    }
    # Load Params
    MODEL_DATA = safe_update_model_data_dict(MODEL_DATA, model_info)
    # Load Model
    MODEL_DATA["config"] = AutoConfig.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"]
    )
    MODEL_DATA["tokenizer"] = AutoTokenizer.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"]
    )
    MODEL_DATA["model"] = AutoModel.from_pretrained(
        HF_ID, 
        cache_dir=MODEL_DATA["hf_params"]["cache_dir"]
    )
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs, **params):
    '''
    HF - Run Model
    '''
    # Init
    TOKENIZER = MODEL_DATA["tokenizer"]
    MODEL = MODEL_DATA["model"]
    CLASSES = None
    try:
        CLASS_KEYS = sorted(list(MODEL_DATA["config"].id2label.keys()))
        CLASSES = [MODEL_DATA["config"].id2label[k] for k in CLASS_KEYS]
    except: pass
    # Run Model
    MODEL_INPUTS = TOKENIZER.batch_encode_plus([inputs["text"]], **MODEL_DATA["params"]["tokenizer"])
    OUTPUTS = MODEL(**MODEL_INPUTS)
    PROB_DIST = OUTPUTS.logits.cpu().detach().numpy()[0]
    CLASS_INDEX = np.argmax(PROB_DIST)
    CLASS = CLASSES[CLASS_INDEX] if CLASSES is not None else f"Class_{CLASS_INDEX}"
    # Form Outputs
    OUTPUTS = {
        "text": inputs["text"],
        "classes": CLASSES,
        "prob_dist": PROB_DIST,
        "class_index": CLASS_INDEX,
        "class": CLASS
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