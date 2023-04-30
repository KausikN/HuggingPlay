"""
HuggingFace - Text - Token Classification - Standard Pipeline

Ref: 
"""

# Imports
from Utils.Utils import *

import torch
from transformers import AutoModelForTokenClassification

import spacy
from spacy import displacy
from spacy.tokens.span import Span as spacy_span

## Import Vars
AutoModel = AutoModelForTokenClassification
try:
    # os.system("python -m spacy download en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")
except Exception as e:
    print(e)
    import en_core_web_sm
    NLP = en_core_web_sm.load()

# Main Functions
## Utils Functions
def Utils_DisplayTokenClassification(tokens, tags, ignore_tags=[], jupyter=False, **params):
    '''
    Utils - Display Token Classification
    '''
    # Generate HTML
    CurDoc = spacy.tokens.Doc(NLP.vocab, words=tokens)
    Ents = []
    for i in range(len(tags)):
        tag = tags[i]
        if tag not in ignore_tags:
            Ents.append(spacy_span(CurDoc, i, i+1, tag))
    CurDoc.set_ents(Ents)
    RENDER_HTML = displacy.render(CurDoc, style="ent", minify=True, jupyter=jupyter)

    return RENDER_HTML

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
    USERINPUT_Inputs["text"] = st.text_area("Enter Text", height=300).strip()
    if USERINPUT_Inputs["text"] == "":
        st.error("Text is empty.")
        st.stop()

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS, **params):
    '''
    UI - Display Outputs
    '''
    # Init
    OUTPUT_PARAMS = OUTPUTS["params"]
    # Save Outputs
    # Display Outputs
    RENDER_HTML = Utils_DisplayTokenClassification(OUTPUTS["text"], OUTPUTS["tags"], **OUTPUT_PARAMS, **params)
    st.write(RENDER_HTML, unsafe_allow_html=True)

## HF Funcs
def HF_Func_LoadModel(model_info, **params):
    '''
    HF - Load Model
    '''
    # Init
    HF_ID = model_info["hf_id"]
    MODEL_DATA = {
        "hf_id": HF_ID,
        "hf_data": model_info["data"],
        "hf_params": {
            "tokenizer": {},
            "model": {},
            "output": {}
        },
        "config": None,
        "tokenizer": None,
        "model": None,
    }
    # Load Params
    if "params" in model_info["data"].keys():
        for k in MODEL_DATA["hf_params"].keys():
            if k in model_info["data"]["params"].keys():
                for pk in model_info["data"]["params"][k].keys():
                    MODEL_DATA["hf_params"][k][pk] = model_info["data"]["params"][k][pk]
    # Load Model
    MODEL_DATA["config"] = AutoConfig.from_pretrained(HF_ID)
    MODEL_DATA["tokenizer"] = AutoTokenizer.from_pretrained(HF_ID)
    MODEL_DATA["model"] = AutoModel.from_pretrained(HF_ID)
    
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
    MODEL_INPUTS = TOKENIZER.batch_encode_plus([inputs["text"]], **MODEL_DATA["hf_params"]["tokenizer"])
    OUTPUTS = MODEL(**MODEL_INPUTS, **MODEL_DATA["hf_params"]["model"])
    PROB_DIST = OUTPUTS.logits.cpu().detach().numpy()[0][1:-1]
    TAGS_CLASS_INDEX = np.argmax(PROB_DIST, axis=-1)
    TAGS = [CLASSES[i] for i in TAGS_CLASS_INDEX] if CLASSES is not None else [f"Class_{i}" for i in TAGS_CLASS_INDEX]
    TOKENS = TOKENIZER.convert_ids_to_tokens(MODEL_INPUTS["input_ids"][0])[1:-1]
    # Form Outputs
    OUTPUTS = {
        "text": inputs["text"],
        "classes": CLASSES,
        "prob_dist": PROB_DIST,
        "tags_class_index": TAGS_CLASS_INDEX,
        "tags": TAGS,
        "tokens": TOKENS,
        "params": MODEL_DATA["hf_params"]["output"]
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