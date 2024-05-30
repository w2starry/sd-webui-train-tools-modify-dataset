# -*- coding: UTF-8 -*-
# The stable diffusion webui training aid extension helps you quickly and visually train models such as Lora.
# repo: https://github.com/liasece/sd-webui-train-tools

from modules import script_callbacks

from liasece_sd_webui_train_tools.ui import *

def on_ui_tabs():
    # init
    
    # the third parameter is the element id on html, with a "tab_" as prefix
    return (new_ui() , "Train LoRA", "train_LoRA"),

script_callbacks.on_ui_tabs(on_ui_tabs)
