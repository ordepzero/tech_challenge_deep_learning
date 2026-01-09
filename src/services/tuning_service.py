
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

# Configuração do logger local
logger = logging.getLogger(__name__)

def tune_model(base_config: TrainRequest, num_samples: int = 10):
    """
    Executa a otimização de hiperparâmetros (tuning) utilizando o Ray Tune.
    
    Args:
        base_config (TrainRequest): Configurações base para o treinamento.
        num_samples (int): Quantidade de amostras (trials) a serem executadas.
        
    Returns:
        dict: Configuração do melhor trial e o ID da run correspondente no MLflow.
    """
    
    logger.info(f"tune_model chamado. Tipo de base_config.model_type: {type(base_config.model_type)}")
    logger.info(f"Valor de base_config.model_type: {base_config.model_type}")

    try:
        model_type_val = getattr(base_config.model_type, "value", str(base_config.model_type))
    except Exception as e:
        logger.error(f"Erro ao obter o valor de model_type: {e}")
        model_type_val = str(base_config.model_type)

    # Define o espaço de busca (search space) baseado no base_config
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "hidden_dim": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout_prob": tune.uniform(0.1, 0.5),
        # Parâmetros fixados
        "model_type": model_type_val,
        "window_size": base_config.window_size,
        "log_level": base_config.log_level
    }

    # Utiliza o algoritmo ASHA para pruning precoce de trials ruins
    scheduler = ASHAScheduler(
        max_t=50,
        grace_period=1,
        reduction_factor=2
    )
    
    def train_func(config_dict):
        """
        Função de treinamento interna para o Ray Tune.
        Adapta o treinamento do PyTorch Lightning para reportar métricas ao Tune.
        """
        # Reconstrói a TrainRequest a partir do dicionário de configuração
        request = TrainRequest(**config_dict)
        
        from src.services.data_loader import StockDataLoader
        from src.services.timeseries_data_module import TimeSeriesDataModule
        from src.models.factory import ModelFactory
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
        from ray.train.lightning import prepare_trainer
        import ray.tune
        import shutil
        from ray.train import Checkpoint

        # Determina o diretório de checkpoint específico dentro do diretório do trial
        trial_dir = os.getcwd()
        ckpt_dir = os.path.join(trial_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"Diretório do Trial: {trial_dir}")
        logger.info(f"Diretório de Checkpoints: {ckpt_dir}")

        class TuneReportCallback(Callback):
            """
            Callback do Lightning para reportar métricas e salvar checkpoints no Ray Tune.
            """
            def on_validation_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                logger.info(f"Chaves de validação: {list(metrics.keys())}") 
                
                report_dict = {}
                for k, v in metrics.items():
                    if hasattr(v, "item"):
                        report_dict[k] = v.item()
                    else:
                        report_dict[k] = v
                
                # Salva manualmente o checkpoint para garantir sua existência
                ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
                trainer.save_checkpoint(ckpt_path)
                
                # Reporta métricas E o checkpoint ao Ray Tune
                ray.tune.report(
                    report_dict, 
                    checkpoint=Checkpoint.from_directory(ckpt_dir)
                )

            def on_train_end(self, trainer, pl_module):
                if os.path.exists(ckpt_dir):
                    logger.info(f"Treinamento finalizado. Checkpoints: {os.listdir(ckpt_dir)}")

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
        
        # Inicializa o Trainer do Lightning com o callback de reporte ao Tune
        trainer = Trainer(
            max_epochs=10,
            accelerator="auto",
            devices="auto",
            enable_progress_bar=False,
            callbacks=[TuneReportCallback()]
        )
        trainer.fit(model, datamodule=data_module)

    # Configura e executa o sintonizador (Tuner) do Ray
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"cpu": 1, "gpu": 0.5} # Aloca recursos fracionários de GPU de forma otimizada
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
    
    logger.info(f"Configuração do melhor trial: {best_result.config}")
    
    mlflow_run_id = None
    
    # Registra o melhor resultado no MLflow em uma nova execução (run)
    # Configura a URI para o banco SQLite para sincronizar com a UI
    if os.path.exists("/app"):
        mlflow.set_tracking_uri("sqlite:////app/mlflow.db")
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("stock_price_prediction")
    with mlflow.start_run(run_name="tuning_best_result") as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_params(best_result.config)
        mlflow.log_metric("best_val_loss", best_result.metrics["val_loss"])
        
        # Localiza e registra o melhor artefato de modelo (checkpoint)
        try:
            if best_result.checkpoint:
                with best_result.checkpoint.as_directory() as checkpoint_dir:
                     ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
                     if ckpt_files:
                        best_ckpt_path = ckpt_files[0]
                        logger.info(f"Melhor checkpoint encontrado em: {best_ckpt_path}")
                        
                        best_config = TrainRequest(**best_result.config)
                        model = ModelFactory.create(best_config)
                        checkpoint = torch.load(best_ckpt_path)
                        model.load_state_dict(checkpoint['state_dict'])
                        
                        mlflow.pytorch.log_model(model, artifact_path="model")
                        logger.info("Melhor modelo registrado no MLflow com sucesso.")
                     else:
                        logger.warning(f"Nenhum arquivo .ckpt encontrado no diretório {checkpoint_dir}")
            else:
                logger.warning("best_result.checkpoint é None. Falha ao recuperar checkpoint do trial.")
                
        except Exception as e:
            logger.error(f"Falha ao registrar o melhor modelo no MLflow: {e}", exc_info=True)

    return {"config": best_result.config, "run_id": mlflow_run_id}