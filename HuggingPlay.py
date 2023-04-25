"""
Set of tools for running and visualizing huggingface models for various tasks
"""

# Imports
from Utils.Utils import *
## Text
from HFLibraries.text.text_classification import HF_text_text_classification
## Image
from HFLibraries.image.depth_estimation import HF_image_depth_estimation
from HFLibraries.image.depth_estimation import HF_image_depth_estimation_glpn
## Audio
from HFLibraries.audio.text_to_speech import HF_audio_text_to_speech
## Multimodal
from HFLibraries.multimodal.text_to_video import HF_multimodal_text_to_video
from HFLibraries.multimodal.text_to_video import HF_multimodal_text_to_video_zero_shot

# Main Functions
def HuggingPlayUtils_LoadModelsInfo(TASKS):
    '''
    HuggingPlay - Utils - Load Models Info
    '''
    # Iterate through all task types
    for task_type in TASKS.keys():
        ## Init
        task_type_str = name_to_path(task_type)
        task_type_path = os.path.join(HF_PATHS["models_data"]["dir"], task_type_str)
        ## Iterate through all tasks
        for task in TASKS[task_type].keys():
            ### Init
            task_str = name_to_path(task)
            task_path = os.path.join(
                task_type_path, 
                HF_PATHS["models_data"]["file"].format(task_type_str=task_type_str, task_str=task_str)
            )
            ### Load Task Data
            task_data = json.load(open(task_path, "r"))
            ### Iterate through all models
            for pk in TASKS[task_type][task].keys(): TASKS[task_type][task][pk]["models"] = {}
            for mk in task_data["models"].keys():
                TASKS[task_type][task][task_data["models"][mk]["pipeline"]]["models"][mk] = task_data["models"][mk]

    return TASKS

# Main Vars
## Path Vars
HF_PATHS = {
    "models_data": {
        "dir": "HFModels/",
        "file": "HFModels_{task_type_str}_{task_str}.json"
    }
}
## Task Vars
TASK_TYPES = list(sorted(os.listdir(HF_PATHS["models_data"]["dir"])))
TASKS = HuggingPlayUtils_LoadModelsInfo({
    "Text": {
        "Text Classification": {
            "default": {
                "module": HF_text_text_classification,
                "models": {}
            }
        }
    },
    "Image": {
        "Depth Estimation": {
            "default": {
                "module": HF_image_depth_estimation,
                "models": {}
            },
            "GLPN": {
                "module": HF_image_depth_estimation_glpn,
                "models": {}
            }
        },
    },
    "Audio": {
        "Text-to-Speech": {
            "default": {
                "module": HF_audio_text_to_speech,
                "models": {}
            }
        }
    },
    "Multimodal": {
        "Text-to-Video": {
            "default": {
                "module": HF_multimodal_text_to_video,
                "models": {}
            },
            "Zero-Shot": {
                "module": HF_multimodal_text_to_video_zero_shot,
                "models": {}
            }
        }
    }
})