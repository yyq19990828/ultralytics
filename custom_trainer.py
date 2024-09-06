from ultralytics.models.yolo.detect.train import DetectionTrainer

import gc
import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)
from ultralytics.nn.tasks import torch_safe_load
from custom_model import CustomModel

class CustomTrainer(DetectionTrainer):


    # def __init__(self, overrides=None):
    #     super().__init__(overrides)
    def save_model(self):
        """Save model 以分段模块的方式"""
        detect = deepcopy(self.ema.ema).half()
        detect.model = detect.model[-1:]
        backbone = deepcopy(self.ema.ema).half()
        backbone.model = backbone.model[:-1]
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'


        # 保存最后一个子模块
        detect_module_buffer = io.BytesIO()
        if self.best_fitness == self.fitness:
            # if isinstance(self.model.detect, CustomModel):
            torch.save({
                'model': detect
                }, 
                detect_module_buffer)
            serialized_detect_module = detect_module_buffer.getvalue()
            # save best_submodule_epoch, i.e. 'detect_module_epoch3.pt'
            (self.wdir / f"detect_module_epoch{self.epoch}.pt").write_bytes(serialized_detect_module)  

        # best_dict, _ = torch_safe_load(f"detect_module_epoch{self.epoch}.pt")
        # print(best_dict['model'])
        # print(type(best_dict))
        # 保存最后一个子模块之前的模块
        backbone_module_buffer = io.BytesIO()
        if self.best_fitness == self.fitness:
            # if isinstance(self.model.backbone, CustomModel):
            torch.save({
                'model': backbone
                }, 
                backbone_module_buffer)
            serialized_backbone_module = backbone_module_buffer.getvalue()
            # save best_submodule_epoch, i.e. 'backbone_module_epoch3.pt'
            (self.wdir / f"backbone_module_epoch{self.epoch}.pt").write_bytes(serialized_backbone_module)

    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        if weights:
            weights = torch.load(weights, map_location='cpu')
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    # nn.Sequential



        