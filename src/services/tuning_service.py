
import ray
from ray import tune
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
import mlflow
from src.schemas.train_request import TrainRequest
from src.services.pipeline_service import train_job, run_train
import logging

logger = logging.getLogger(__name__)

def tune_model(base_config: TrainRequest, num_samples: int = 10):
    """
    Run hyperparameter tuning using Ray Tune.
    """
    
    # Define search space based on base_config
    # This is a simple example; in a real scenario, the search space might be passed in
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "hidden_dim": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout_prob": tune.uniform(0.1, 0.5),
        # Keep other params fixed
        "model_type": base_config.model_type,
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
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from ray.train.lightning import RayTrainReportCallback, prepare_trainer
        
        base_path = "."
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
            callbacks=[RayTrainReportCallback()]
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=data_module)
        # END inline adaptation

    import os
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"cpu": 1, "gpu": 0.5} # Assuming we want to parallelize
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
            callbacks=[] 
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    logger.info(f"Best trial config: {best_result.config}")
    
    # Log best result to MLflow as a separate run or tag
    mlflow.set_experiment("stock_price_prediction")
    with mlflow.start_run(run_name="tuning_best_result"):
        mlflow.log_params(best_result.config)
        mlflow.log_metric("best_val_loss", best_result.metrics["val_loss"])
        
    return best_result.config
