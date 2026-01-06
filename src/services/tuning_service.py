
import ray
from ray import tune
from ray.tune import RunConfig
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
import mlflow
from src.schemas.train_request import TrainRequest
from src.services.pipeline_service import train_job, run_train
from src.models.factory import ModelFactory
import logging
import os
import glob
import torch

logger = logging.getLogger(__name__)

def tune_model(base_config: TrainRequest, num_samples: int = 10):
    """
    Run hyperparameter tuning using Ray Tune.
    """
    
    logger.info(f"Tune Model called. base_config.model_type type: {type(base_config.model_type)}")
    logger.info(f"base_config.model_type value: {base_config.model_type}")

    try:
        model_type_val = getattr(base_config.model_type, "value", str(base_config.model_type))
    except Exception as e:
        logger.error(f"Error getting model_type value: {e}")
        model_type_val = str(base_config.model_type)

    # Define search space based on base_config
    # This is a simple example; in a real scenario, the search space might be passed in
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "hidden_dim": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout_prob": tune.uniform(0.1, 0.5),
        # Keep other params fixed
        "model_type": model_type_val,
        "window_size": base_config.window_size,
        "log_level": base_config.log_level
    }

    scheduler = ASHAScheduler(
        max_t=50,
        grace_period=1,
        reduction_factor=2
    )
    
    # Define standard training function wrapper for Ray Tune
    def train_func(config_dict):
        # Reconstruct TrainRequest from config_dict
        request = TrainRequest(**config_dict)
        # We need to adapt run_train to work with Ray Tune reporting
        # For simplicity, we are calling the existing run_train logic
        # Ideally, run_train should report metrics to Ray Tune
        
        # NOTE: This requires refactoring run_train or using a specific Ray Train API
        # Since run_train uses Lightning Trainer, we can integrate Ray Train Lightning
        
        # START inline adaptation of run_train for tuning context
        from src.services.data_loader import StockDataLoader
        from src.services.timeseries_data_module import TimeSeriesDataModule
        from src.models.factory import ModelFactory
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
        from ray.train.lightning import prepare_trainer
        import ray.tune
        import shutil
        from ray.train import Checkpoint

        # Determine explicit checkpoint directory in the current working dir (Trial dir)
        trial_dir = os.getcwd()
        ckpt_dir = os.path.join(trial_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"Trial Directory: {trial_dir}")
        logger.info(f"Checkpoint Directory: {ckpt_dir}")

        class TuneReportCallback(Callback):
            def on_validation_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                logger.info(f"Validation keys: {list(metrics.keys())}") 
                
                report_dict = {}
                for k, v in metrics.items():
                    if hasattr(v, "item"):
                        report_dict[k] = v.item()
                    else:
                        report_dict[k] = v
                
                # Manually save checkpoint to ensure it exists
                ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
                trainer.save_checkpoint(ckpt_path)
                
                # Report metrics AND checkpoint to Ray Tune
                ray.tune.report(
                    report_dict, 
                    checkpoint=Checkpoint.from_directory(ckpt_dir)
                )

            def on_train_end(self, trainer, pl_module):
                if os.path.exists(ckpt_dir):
                    logger.info(f"Train end. Checkpoints: {os.listdir(ckpt_dir)}")

        base_path = "/app" if os.path.exists("/app") else "."
        loader = StockDataLoader(raw_path=f"{base_path}/data/raw", processed_path=f"{base_path}/data/processed")
        filename_path = loader.get_latest_file_path("NVDA", kind="raw")
        
        data_module = TimeSeriesDataModule(
            csv_path=filename_path,
            value_col="Close",
            window_size=request.window_size,
            batch_size=32
        )
        data_module.setup()
        model = ModelFactory.create(request)
        
        trainer = Trainer(
            max_epochs=10,
            accelerator="auto",
            devices="auto",
            enable_progress_bar=False,
            callbacks=[TuneReportCallback()]
        )
        trainer.fit(model, datamodule=data_module)
        # END inline adaptation

    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"cpu": 1, "gpu": 0.5} 
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=RunConfig(
            name="tune_stock_prediction",
            storage_path=os.path.abspath("./ray_results") if not os.path.exists("/app") else "/app/ray_results",
            verbose=1,
            callbacks=[] 
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    logger.info(f"Best trial config: {best_result.config}")
    
    mlflow_run_id = None
    
    # Log best result to MLflow as a separate run or tag
    mlflow.set_experiment("stock_price_prediction")
    with mlflow.start_run(run_name="tuning_best_result") as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_params(best_result.config)
        mlflow.log_metric("best_val_loss", best_result.metrics["val_loss"])
        
        # Locate and log the best model artifact
        try:
            if best_result.checkpoint:
                with best_result.checkpoint.as_directory() as checkpoint_dir:
                     ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
                     if ckpt_files:
                        best_ckpt_path = ckpt_files[0]
                        logger.info(f"Found best checkpoint at: {best_ckpt_path}")
                        
                        best_config = TrainRequest(**best_result.config)
                        model = ModelFactory.create(best_config)
                        checkpoint = torch.load(best_ckpt_path)
                        model.load_state_dict(checkpoint['state_dict'])
                        
                        mlflow.pytorch.log_model(model, artifact_path="model")
                        logger.info("Best model logged to MLflow successfully.")
                     else:
                        logger.warning(f"No .ckpt files found in checkpoint dir {checkpoint_dir}")
            else:
                logger.warning("best_result.checkpoint is None. Falling back to path search...")
                
        except Exception as e:
            logger.error(f"Failed to log best model to MLflow: {e}", exc_info=True)

    return {"config": best_result.config, "run_id": mlflow_run_id}
