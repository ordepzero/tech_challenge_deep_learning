
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any, Optional
import os

class MLFlowManager:
    """
    Gerencia a integração com o MLflow para rastreamento de experimentos,
    registro de modelos e gerenciamento de artefatos.
    """
    def __init__(self, experiment_name: str = "stock_price_prediction"):
        """
        Inicializa o MLFlowManager, configura a URI de rastreamento e o experimento.
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Define a URI de rastreamento para usar o banco de dados SQLite sincronizado com a UI
        if os.path.exists("/app"):
            tracking_uri = "sqlite:////app/mlflow.db"
        else:
            tracking_uri = "sqlite:///mlflow.db"
            
        mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)

    def list_runs(self) -> List[Dict[str, Any]]:
        """
        Lista todas as execuções (runs) do experimento atual, ordenadas por tempo de início.
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        # Converte o DataFrame para uma lista de dicionários para facilitar o consumo consumos
        runs_list = []
        if not runs.empty:
            for _, run in runs.iterrows():
                # Remove valores NaN para evitar problemas na serialização JSON
                run_data = run.dropna().to_dict()
                runs_list.append(run_data)
        
        return runs_list

    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """
        Obtém informações detalhadas de uma execução específica via run_id.
        """
        run = mlflow.get_run(run_id)
        return {
            "info": dict(run.info),
            "data": dict(run.data),
        }

    def load_model(self, run_id: str):
        """
        Carrega um modelo PyTorch a partir de uma execução específica.
        """
        model_uri = f"runs:/{run_id}/model"
        return mlflow.pytorch.load_model(model_uri)

    def log_input_data(self, run_id: str, data_path: str):
        """
        Registra informações sobre a fonte de dados de entrada em uma execução.
        """
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param("data_source", data_path)

    def search_best_model(self, metric: str = "val_loss", mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        Busca a melhor execução baseada em uma métrica e modo (min/max).
        """
        order = f"metrics.{metric} ASC" if mode == "min" else f"metrics.{metric} DESC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[order],
            max_results=1
        )
        
        if not runs.empty:
            return runs.iloc[0].to_dict()
        return None

