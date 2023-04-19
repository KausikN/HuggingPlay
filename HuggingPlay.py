"""
Set of tools for running and visualizing huggingface models for various tasks
"""

# Imports
from Utils.Utils import *
## Text
## Image
## Audio
## Multimodal
from HFLibraries.Multimodal import HF_Multimodal_Text_to_Video

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
            TASKS[task_type][task]["models"] = task_data["models"]

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
            "module": None,
            "models": {}
        }
    },
    "Image": {
        "Depth Estimation": {
            "module": None,
            "models": {}
        },
    },
    "Audio": {
        "Text-to-Speech": {
            "module": None,
            "models": {}
        }
    },
    "Multimodal": {
        "Text-to-Video": {
            "module": HF_Multimodal_Text_to_Video,
            "models": {}
        }
    }
})