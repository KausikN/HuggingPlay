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
        USERINPUT_ModelInfo["data"] = json.loads(st.text_area(
            "Enter Model Data (JSON)",
            json.dumps(TASK_DATA["models"][USERINPUT_ModelInfo["hf_id"]], indent=8),
            height=300
        ))
    ## Custom HF-ID
    else:
        USERINPUT_ModelInfo["hf_id"] = st.text_input("Enter Model HF-ID").strip()
        if USERINPUT_ModelInfo["hf_id"] == "":
            st.error("Invalid HF-ID.")
            st.stop()
        DefaultParams = TASK_DATA["models"][list(TASK_DATA["models"].keys())[0]]
        USERINPUT_ModelInfo["data"] = json.loads(st.text_area(
            "Enter Model Data (JSON)",
            json.dumps(DefaultParams, indent=8),
            height=300
        ))

    return USERINPUT_ModelInfo

def UI_LoadAgent(USERINPUT_AgentType):
    '''
    UI - Load Agent
    '''
    st.markdown("## Load Agent")
    # Init
    USERINPUT_AgentInfo = {
        "token": "",
        "data": {}
    }
    # Enter Token
    if USERINPUT_AgentType == "OpenAI":
        USERINPUT_AgentInfo["token"] = st.text_input("Enter OpenAI API Key").strip()
    else:
        USERINPUT_AgentInfo["token"] = st.text_input("Enter HF Token").strip()

    return USERINPUT_AgentInfo

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
    LoadCache()

    # Load Inputs
    ## Load Task
    USERINPUT_TaskType = st.sidebar.selectbox("Task Type", TASKS.keys())
    USERINPUT_Task = st.sidebar.selectbox("Task", TASKS[USERINPUT_TaskType].keys())
    USERINPUT_Pipeline = st.sidebar.selectbox("Pipeline", TASKS[USERINPUT_TaskType][USERINPUT_Task].keys())
    TASK_DATA = dict(TASKS[USERINPUT_TaskType][USERINPUT_Task][USERINPUT_Pipeline])
    TASK_DATA.update({
        "task_type": USERINPUT_TaskType,
        "task": USERINPUT_Task,
        "pipeline": USERINPUT_Pipeline
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
    OUTPUTS = MODULE.HF_FUNCS["run_model"](MODEL_DATA, USERINPUT_Inputs["inputs"])

    # Display Outputs
    st.markdown("## Outputs")
    MODULE.UI_FUNCS["display_outputs"](OUTPUTS, **CACHE["settings"])

def run_agent():
    # Title
    st.header("HuggingPlay - Agent")

    # Prereq Loaders
    LoadCache()

    # Load Inputs
    ## Load Agent
    USERINPUT_AgentType = st.sidebar.selectbox("Agent Type", AGENTS.keys())
    AGENT_DATA = AGENTS[USERINPUT_AgentType]
    USERINPUT_AgentInfo = UI_LoadAgent(USERINPUT_AgentType)
    ## Load Inputs
    USERINPUT_Inputs = UI_LoadInputs(AGENT_DATA)

    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Process Inputs
    MODULE = AGENT_DATA["module"]
    ## Load Model
    MODEL_DATA = MODULE.HF_FUNCS["load_agent"](USERINPUT_AgentInfo)
    ## Run Model using Inputs
    OUTPUTS = MODULE.HF_FUNCS["run_agent"](MODEL_DATA, USERINPUT_Inputs["inputs"])

    # Display Outputs
    st.markdown("## Outputs")
    MODULE.UI_FUNCS["display_outputs"](OUTPUTS, **CACHE["settings"])

def settings():
    global CACHE
    # Title
    st.header("Settings")

    # Load Inputs
    ## Init
    LoadCache()
    SETTINGS = CACHE["settings"] if "settings" in CACHE.keys() else {
        "interactive_display": True
    }
    ## Plots Settings
    st.markdown("## Plots Settings")
    ### Interactive Plots
    SETTINGS["interactive_display"] = st.checkbox("Interactive Display", value=SETTINGS["interactive_display"])
    # Save Inputs
    if st.button("Save Settings"):
        CACHE["settings"] = SETTINGS
        SaveCache()
        st.success("Settings Saved.")

    ## Operations
    st.markdown("## Operations")
    ### Clear HF Cache
    if st.button("Clear HF Cache"):
        os.system(f"rm -rf {HF_CACHE_PATH}")
        os.makedirs(HF_CACHE_PATH, exist_ok=True)
        st.success("Hugging-Face Cache Cleared.")
    ### Set Local HF Cache Path
    HF_CACHE_KEY = "default" if st.checkbox("Default HF Cache Path") else "local"
    for k in CACHE["hf_cache_env_vars"].keys(): os.environ[k] = CACHE["hf_cache_env_vars"][k][HF_CACHE_KEY]
        

#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()