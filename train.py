from os.path import basename, join
import os
import torch
from commode_utils.callback import PrintEpochResultCallback, UploadCheckpointCallback
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


def train(model: LightningModule, data_module: LightningDataModule,
          config: DictConfig):
    # Define logger
    model_name = model.__class__.__name__
    dataset_name = basename(config.dataset.name)
    # tensorboard logger
    # 创建 TensorBoardLogger 对象，用于记录训练过程中的日志。日志将保存在指定的目录中，其中包含模型名称和数据集名称。
    tensorlogger = TensorBoardLogger(join("ts_logger", model_name),
                                     dataset_name)
    # define model checkpoint callback
    # 创建 ModelCheckpoint 对象，用于保存训练过程中的模型检查点。检查点将保存在指定的目录中，并根据验证集上的损失值进行监控和命名。
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(tensorlogger.log_dir, "checkpoints"),
        monitor="val_loss",
        filename="{epoch:02d}-{step:02d}-{val_loss:.4f}",
        every_n_val_epochs=1,
        save_top_k=5,
    )

    # 创建 UploadCheckpointCallback 对象，用于将模型权重上传到指定的位置。
    upload_weights = UploadCheckpointCallback(
        join(tensorlogger.log_dir, "checkpoints"))

    # 创建 EarlyStopping 对象，用于在验证集上监测损失值，并在指定的轮数内没有改进时提前停止训练。
    early_stopping_callback = EarlyStopping(patience=config.hyper_parameters.patience,
                                            monitor="val_loss",
                                            verbose=True,
                                            mode="min")

    # 创建 LearningRateMonitor 对象，用于记录学习率的变化情况。
    lr_logger = LearningRateMonitor("step")
    # 创建 PrintEpochResultCallback 对象，用于在每个 epoch 结束时打印结果。
    print_epoch_results = PrintEpochResultCallback(split_symbol="_",
                                                   after_test=False)

    # 根据是否有可用的 GPU，设置 gpu 变量为 1 或 None
    gpu = 1 if torch.cuda.is_available() else None
    # 创建 Trainer 对象，用于配置训练过程的参数和回调函数。设置最大训练轮数、梯度裁剪阈值、是否使用确定性计算等。
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        val_check_interval=config.hyper_parameters.val_every_step,
        log_every_n_steps=config.hyper_parameters.log_every_n_steps,
        logger=[tensorlogger],
        gpus=gpu,
        progress_bar_refresh_rate=config.hyper_parameters.progress_bar_refresh_rate,
        callbacks=[
            lr_logger, early_stopping_callback, checkpoint_callback,
            print_epoch_results, upload_weights
        ],
        resume_from_checkpoint=config.hyper_parameters.resume_from_checkpoint,
    )

    # 调用 Trainer 的 fit 方法开始训练，传入模型和数据模块作为参数。
    # fit() 方法会在每个 epoch 结束后执行验证，并在训练过程中记录指标。它还会根据需要执行检查点保存、提前停止等操作。
    trainer.fit(model=model, datamodule=data_module)
    # 调用 Trainer 的 test 方法进行模型测试，传入模型作为参数。
    trainer.test(model=model)
