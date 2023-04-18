"""
Set of tools for running and visualizing huggingface models for various tasks
"""

# Imports
## Text
## Image
## Audio
## Multimodal
from HFLibraries.Multimodal import HF_Multimodal_Text2Video

# Main Vars
MODULES = {
    "text": {},
    "image": {},
    "audio": {},
    "multimodal": {
        "text2video": HF_Multimodal_Text2Video
    }
}

# Main Functions