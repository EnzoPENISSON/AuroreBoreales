import os
from pytorch_lightning.strategies import DDPStrategy

def create_model(n_epochs: int = N_EPOCHS, force_reset: bool = True) -> TFTModel:
    accelerator_cfg = _get_accelerator_config()

    # Read environment variables set by the PyTorch launcher
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # LOCAL_RANK identifies the specific process on the current machine
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # We MUST use 'gloo' because 'nccl' does not support CPU or MPS
        ddp_strategy = DDPStrategy(process_group_backend="gloo")
        ddp_cfg = {
            "num_nodes": world_size,
            "strategy": ddp_strategy,
            "sync_batchnorm": False, # Important to disable for mixed hardware
        }
    else:
        ddp_cfg = {}

    # ---------------------------------------------------------
    # Logging and Callbacks: Only initialize on Rank 0
    # ---------------------------------------------------------
    loggers = []
    if local_rank == 0:
        loggers.append(CSVLogger(save_dir=str(MODEL_DIR / "logs"), name="aurora_tft"))

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        mode="min",
        verbose=(local_rank == 0), # Only print on Master
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    callbacks = [early_stopper, lr_monitor]
    if local_rank == 0:
        callbacks.append(EpochStatsPrinter())

    return TFTModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        num_attention_heads=ATTENTION_HEADS,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": LR},
        add_relative_index=True,
        random_state=42,
        model_name="aurora_tft",
        work_dir=str(MODEL_DIR),
        save_checkpoints=True,
        force_reset=force_reset,
        pl_trainer_kwargs={
            **accelerator_cfg,
            **ddp_cfg,
            "callbacks": callbacks,
            "logger": loggers if loggers else False,
            "log_every_n_steps": 10,
            "enable_progress_bar": (local_rank == 0), # Clean terminal output
        },
    )
