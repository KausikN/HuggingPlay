"""
HuggingFace - Audio - Text to Speech - Standard Pipeline (Speech T5)

Ref: 
"""

# Imports
from Utils.Utils import *

import soundfile as sf

import torch
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Main Functions
## UI Funcs
def UI_Func_LoadInputs():
    '''
    UI - Load Inputs
    '''
    # Init
    USERINPUT_Inputs = {
        "processor": {
            "text": ""
        }
    }
    # Ask Inputs
    ## Prompt
    USERINPUT_Inputs["processor"]["text"] = st.text_input("Enter Text").strip()
    if USERINPUT_Inputs["processor"]["text"] == "":
        st.error("No text provided.")
        st.stop()

    return USERINPUT_Inputs

def UI_Func_DisplayOutputs(OUTPUTS):
    '''
    UI - Display Outputs
    '''
    # Init
    AUDIO_DATA = OUTPUTS["audio_data"]
    AUDIO_SAVE_PATH = os.path.join(UTILS_PATHS["temp"], "HF_audio_text_to_speech.wav")
    # Save Outputs
    sf.write(AUDIO_SAVE_PATH, AUDIO_DATA, samplerate=16000)
    # Display Outputs
    st.audio(AUDIO_SAVE_PATH)

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
            "speaker": {
                "vocoder_id": "microsoft/speecht5_hifigan",
                "dataset_id": "Matthijs/cmu-arctic-xvectors",
                "split": "validation",
                "xvector_column": "xvector",
                "xvector_index": 0,
            }
        },
        "processor": None,
        "model": None,
        "vocoder": None,
        "voice_embeddings": None
    }
    # Load Params
    if "params" in model_info["data"].keys():
        for k in MODEL_DATA["hf_params"].keys():
            if k in model_info["data"]["params"].keys():
                for pk in model_info["data"]["params"][k].keys():
                    MODEL_DATA["hf_params"][k][pk] = model_info["data"]["params"][k][pk]
    # Load Model
    MODEL_DATA["processor"] = SpeechT5Processor.from_pretrained(HF_ID)
    MODEL_DATA["model"] = SpeechT5ForTextToSpeech.from_pretrained(HF_ID)
    MODEL_DATA["vocoder"] = SpeechT5HifiGan.from_pretrained(MODEL_DATA["hf_params"]["speaker"]["vocoder_id"])
    #  Load xvector containing speaker's voice characteristics from a dataset
    voc_emb_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    MODEL_DATA["voice_embeddings"] = torch.tensor(
        voc_emb_dataset[MODEL_DATA["hf_params"]["speaker"]["xvector_index"]][MODEL_DATA["hf_params"]["speaker"]["xvector_column"]]
    ).unsqueeze(0)
    
    return MODEL_DATA

def HF_Func_RunModel(MODEL_DATA, inputs):
    '''
    HF - Run Model
    '''
    # Init
    MODEL = MODEL_DATA["model"]
    # Run Model
    PROC_INPUTS = MODEL_DATA["processor"](text=inputs["processor"]["text"], **MODEL_DATA["hf_params"]["processor"])
    AUDIO_DATA = MODEL.generate_speech(PROC_INPUTS["input_ids"], MODEL_DATA["voice_embeddings"], vocoder=MODEL_DATA["vocoder"])
    AUDIO_DATA = AUDIO_DATA.numpy()
    OUTPUTS = {
        "audio_data": AUDIO_DATA
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