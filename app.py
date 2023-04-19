"""
Stream lit GUI for hosting HuggingPlay
"""

# Imports
import os
import json
import streamlit as st

from HuggingPlay import *

# Main Vars
config = json.load(open("./StreamLitGUI/UIConfig.json", "r"))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    "Choose one of the following",
        tuple(
            [config["PROJECT_NAME"]] + 
            config["PROJECT_MODES"]
        )
    )
    
    if selected_box == config["PROJECT_NAME"]:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(" ", "_").lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config["PROJECT_NAME"])
    st.markdown("Github Repo: " + "[" + config["PROJECT_LINK"] + "](" + config["PROJECT_LINK"] + ")")
    st.markdown(config["PROJECT_DESC"])

    # st.write(open(config["PROJECT_README"], "r").read())

#############################################################################################################################
# Repo Based Vars
PATHS = {
    "cache": "StreamLitGUI/CacheData/Cache.json",
}

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(PATHS["cache"], "r"))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(PATHS["cache"], "w"), indent=4)

# Main Functions


# UI Functions
def UI_LoadModel(TASK_DATA):
    '''
    UI - Load Model
    '''
    st.markdown("## Load Model")
    # Init
    USERINPUT_ModelInfo = {
        "hf_id": "",
        "data": {}
    }
    # Select Load Type
    USERINPUT_ModelLoadType = st.selectbox("Mode", ["Available HF-ID", "Custom HF-ID"])
    ## Available HF-ID
    if USERINPUT_ModelLoadType == "Available HF-ID":
        USERINPUT_ModelInfo["hf_id"] = st.selectbox("Select Model", list(TASK_DATA["models"].keys()))
        if USERINPUT_ModelInfo["hf_id"] == None:
            st.error("No Documented Models Available for this Task.")
            st.stop()
        USERINPUT_ModelInfo["data"] = TASK_DATA["models"][USERINPUT_ModelInfo["hf_id"]]
    ## Custom HF-ID
    else:
        USERINPUT_ModelInfo["hf_id"] = st.text_input("Enter Model HF-ID").strip()
        if USERINPUT_ModelInfo["hf_id"] == "":
            st.error("Invalid HF-ID.")
            st.stop()

    return USERINPUT_ModelInfo

def UI_LoadInputs(TASK_DATA):
    '''
    UI - Load Inputs
    '''
    st.markdown("## Load Inputs")
    # Init
    MODULE = TASK_DATA["module"]
    USERINPUT_Inputs = {
        "inputs": {}
    }
    # Ask Inputs for Task
    USERINPUT_Inputs["inputs"] = MODULE.UI_FUNCS["load_inputs"]()

    return USERINPUT_Inputs

# Repo Based Functions
def run_models():
    # Title
    st.header("HuggingPlay - Run")

    # Prereq Loaders

    # Load Inputs
    ## Load Task
    USERINPUT_TaskType = st.sidebar.selectbox("Task Type", TASKS.keys())
    USERINPUT_Task = st.sidebar.selectbox("Task", TASKS[USERINPUT_TaskType].keys())
    TASK_DATA = dict(TASKS[USERINPUT_TaskType][USERINPUT_Task])
    TASK_DATA.update({
        "task_type": USERINPUT_TaskType,
        "task": USERINPUT_Task
    })
    st.markdown(f"## {USERINPUT_TaskType} - {USERINPUT_Task}")
    ## Load Model
    USERINPUT_ModelInfo = UI_LoadModel(TASK_DATA)
    ## Load Inputs
    USERINPUT_Inputs = UI_LoadInputs(TASK_DATA)

    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Process Inputs
    MODULE = TASK_DATA["module"]
    ## Load Model
    MODEL_DATA = MODULE.HF_FUNCS["load_model"](USERINPUT_ModelInfo)
    ## Run Model using Inputs
    OUTPUTS = MODULE.HF_FUNCS["run_model"](MODEL_DATA, USERINPUT_Inputs)

    # Display Outputs
    st.markdown("## Outputs")
    MODULE.UI_FUNCS["display_outputs"](OUTPUTS)
    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()