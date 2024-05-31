# -*- coding: UTF-8 -*-

import gradio as gr
import os
import sys
import traceback
from modules import shared
from modules import sd_models
from liasece_sd_webui_train_tools.util import *
from liasece_sd_webui_train_tools import train
from liasece_sd_webui_train_tools import ArgsList
import numpy as np

from liasece_sd_webui_train_tools.project import *
from liasece_sd_webui_train_tools.config_file import *
# from liasece_sd_webui_train_tools.project_version_ui import *
# from liasece_sd_webui_train_tools.ui import *
from liasece_sd_webui_train_tools.checkpoint_preview_ui import *
# from liasece_sd_webui_train_tools.dateset_ui import *
# from liasece_sd_webui_train_tools.train_ui import *
import liasece_sd_webui_train_tools.sd_scripts.train_network as train_network

def model_path_for_train():
    return shared.cmd_opts.ckpt_dir if shared.cmd_opts.ckpt_dir is not None else sd_models.model_path

def on_train_begin_click(id: str, project: str, version: str,
        # train config
        train_base_model: str, 
        train_batch_size: int, 
        train_num_epochs: int, 
        train_save_every_n_epochs: int,
        train_finish_generate_all_checkpoint_preview: bool,
        train_optimizer_type: str,
        train_learning_rate: str,
        sd_script_args: str,
        train_net_dim: int,
        train_alpha: int,
        train_clip_skip: int,
        train_mixed_precision: str,
        train_xformers: bool,
        train_base_on_sd_v2: bool,
        use_sdxl: bool, # use sdxl
        train_scheduler: str,
        train_cosine_restarts: str,
        train_polynomial_power: str,
        train_unet_lr: str,
        train_text_encoder_lr: str,
        # preview view config
        preview_include_sub_img: bool,
        # txt2txt
        preview_txt2img_prompt: str, # like "apple"
        preview_txt2img_negative_prompt: str, # like "apple"
        preview_sampling_method: list[str], # like `"Euler a", "ms"`
        preview_sampling_steps: str, # like 20,24,28
        preview_width: int, # like 512
        preview_height: int, # like 512
        preview_batch_count: int, # like 1
        preview_batch_size: int, # like 1
        preview_cfg_scale: str, # like 8,9,10,11
        preview_seed: str, # like -1,-1
        preview_lora_multiplier: str, # like 0.6,0.7,0.8,0.9
    ):

    # train_network.train_judge = True
    train_network.current_train.value = 1
    
    train_learning_rate = float(train_learning_rate)
    if train_unet_lr == "":
        train_unet_lr = None
    else:
        train_unet_lr = float(train_unet_lr)

    if train_text_encoder_lr == "":
        train_text_encoder_lr = None
    else:
        train_text_encoder_lr = float(train_text_encoder_lr)

    if train_cosine_restarts == "":
        train_cosine_restarts = None
    else:
        train_cosine_restarts = int(train_cosine_restarts)

    if train_polynomial_power == "":
        train_polynomial_power = None
    else:
        train_polynomial_power = float(train_polynomial_power)
        
    save_train_config(project, version, {
        # train config
        "train_base_model": train_base_model,
        "train_batch_size": int(train_batch_size),
        "train_num_epochs": int(train_num_epochs),
        "train_save_every_n_epochs": int(train_save_every_n_epochs),
        "train_finish_generate_all_checkpoint_preview": train_finish_generate_all_checkpoint_preview,
        "train_optimizer_type": train_optimizer_type,
        "train_learning_rate": float(train_learning_rate),
        "sd_script_args": sd_script_args,
        "train_net_dim": int(train_net_dim),
        "train_alpha": int(train_alpha),
        "train_clip_skip": int(train_clip_skip),
        "train_mixed_precision": train_mixed_precision,
        "train_xformers": train_xformers,
        "train_base_on_sd_v2": train_base_on_sd_v2,
        "use_sdxl": use_sdxl,
        "train_scheduler": train_scheduler,
        "train_cosine_restarts": train_cosine_restarts,
        "train_polynomial_power": train_polynomial_power,
        "train_unet_lr": train_unet_lr,
        "train_text_encoder_lr": train_text_encoder_lr,
    })
    save_preview_config(project, version, {
        # preview view config
        "preview_include_sub_img": preview_include_sub_img,
        # txt2txt
        "preview_txt2img_prompt": preview_txt2img_prompt, # like "apple"
        "preview_txt2img_negative_prompt": preview_txt2img_negative_prompt, # like "apple"
        "preview_sampling_method": preview_sampling_method, # like `"Euler a", "ms"`
        "preview_sampling_steps": preview_sampling_steps, # like 20,24,28
        "preview_width": preview_width, # like 512
        "preview_height": preview_height, # like 512
        "preview_batch_count": preview_batch_count, # like 1
        "preview_batch_size": preview_batch_size, # like 1
        "preview_cfg_scale": preview_cfg_scale, # like 8,9,10,11
        "preview_seed": preview_seed, # like -1,-1
        "preview_lora_multiplier": preview_lora_multiplier, # like 0.6,0.7,0.8,0.9
    })
    train_base_model_path = ""
    train_base_model_name = ""
    for x in sd_models.checkpoints_list.values():
        if x.title == train_base_model:
            train_base_model_path = os.path.join(model_path_for_train(), x.name)
            train_base_model_name = os.path.splitext(x.name)[0]
            break
    processed_path = get_project_version_dataset_processed_path(project, version)
    os.makedirs(processed_path, exist_ok=True)


    if train_base_model_name != "":
        train_name = (train_base_model_name + "-bs-" + str(int(train_batch_size)) + "-ep-" + str(
            int(train_num_epochs)) + "-op-" + str(train_optimizer_type) + "-lr-" + str(
            float(train_learning_rate)) + "-net-" + str(int(train_net_dim)) + "-ap-" + str(
            int(train_alpha))).replace(" ", "").replace(".", "_")
        checkpoint_save_path = get_project_version_trains_checkpoint_path(project, version, train_name)
        os.makedirs(checkpoint_save_path, exist_ok=True)
        cfg = ArgsList.ArgStore()  # ArgsList中定义的类名，可能涉及后端train_config的修改，用于真正训练中的参数配置
        cfg.img_folder = os.path.abspath(processed_path)  # 数据集
        cfg.output_folder = os.path.abspath(checkpoint_save_path)
        cfg.change_output_name = project + r"-" + version
        cfg.batch_size = int(train_batch_size)
        cfg.num_epochs = int(train_num_epochs)
        cfg.save_every_n_epochs = int(train_save_every_n_epochs)
        cfg.base_model = train_base_model_path
        cfg.optimizer_type = train_optimizer_type
        cfg.learning_rate = float(train_learning_rate)
        cfg.unet_lr = train_unet_lr
        cfg.text_encoder_lr = train_text_encoder_lr
        cfg.net_dim = int(train_net_dim)
        cfg.alpha = int(train_alpha)
        cfg.clip_skip = int(train_clip_skip)
        cfg.mixed_precision = train_mixed_precision
        cfg.xformers = train_xformers
        cfg.v2 = train_base_on_sd_v2
        cfg.use_sdxl = use_sdxl
        cfg.ext_sd_script_args = sd_script_args
        cfg.scheduler = train_scheduler
        cfg.cosine_restarts = train_cosine_restarts
        cfg.scheduler_power = train_polynomial_power
        # check if reg path exist
        if os.path.exists(os.path.join(processed_path, "..", "reg")):
            cfg.reg_img_folder = os.path.abspath(os.path.join(processed_path, "..", "reg"))
        printD("on_train_begin_click", cfg.__dict__)
        try:
            train.train(cfg)  # 调用liasece下的train.py中的train函数-->然后再调用sd_scripts下的train_network，真正开始训练
        except Exception as e:
            printD("train.train error", e)
            print(traceback.format_exc(), file=sys.stderr)

    
    # generate preview
    if train_finish_generate_all_checkpoint_preview and train_network.current_train.value == 1:
        return [None]+on_ui_preview_generate_all_preview_btn_click(id, project, version, train_name,
            preview_include_sub_img,
            preview_txt2img_prompt,
            preview_txt2img_negative_prompt,
            preview_sampling_method,
            preview_sampling_steps,
            preview_width,
            preview_height,
            preview_batch_count,
            preview_batch_size,
            preview_cfg_scale,
            preview_seed,
            preview_lora_multiplier,
        )
    return [None]+gr_update_trains_area_list(project, version, train_name)+["done"]

def train_base_model_refresh_click():
    shared.refresh_checkpoints()
    tiles = shared.list_checkpoint_tiles()
    return [
        gr.Dropdown.update(visible=True, choices=tiles,value = tiles[0] if len(tiles) > 0 else ""),
    ]
