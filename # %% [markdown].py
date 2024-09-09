# %%
from custom_trainer import CustomTrainer

dict = {
    'model': '/home/tyjt/桌面/ultralytics/custom_model_config/yolov8-ghost-onlyp2.yaml',
    'data': 'coco8.yaml',
    'epochs': 1,
    'save': False,
    # 'pretrained': '/home/tyjt/桌面/ultralytics/runs/detect/train8/weights/detect_module_epoch0.pt'
}
trainer = CustomTrainer(overrides=dict)
# trainer.setup_model()
# print(trainer.model.names)
# trainer.get_model(cfg=trainer.model, weights=trainer.args.pretrained)


# # %%
# trainer.model.model[-1]

# %%
trainer.train()


