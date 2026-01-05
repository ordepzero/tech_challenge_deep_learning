
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any, Optional
import os

class MLFlowManager:
    def __init__(self, experiment_name: str = "stock_price_prediction"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        tracking_uri = "file:///app/mlruns" if os.path.exists("/app") else "file:./mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs for the current experiment."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        # Convert DataFrame to list of dicts for easier consumption
        runs_list = []
        if not runs.empty:
            for _, run in runs.iterrows():
                # Filter out NaN values to avoid JSON serialization issues
                run_data = run.dropna().to_dict()
                runs_list.append(run_data)
        
        return runs_list

    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run."""
        run = mlflow.get_run(run_id)
        return {
            "info": dict(run.info),
            "data": dict(run.data),
        }

    def load_model(self, run_id: str):
        """Load a PyTorch model from a specific run."""
        model_uri = f"runs:/{run_id}/model"
        return mlflow.pytorch.load_model(model_uri)

    def log_input_data(self, run_id: str, data_path: str):
        """Log input data source information."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param("data_source", data_path)

    def search_best_model(self, metric: str = "val_loss", mode: str = "min") -> Optional[Dict[str, Any]]:
        """Find the best model based on a metric."""
        order = f"metrics.{metric} ASC" if mode == "min" else f"metrics.{metric} DESC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[order],
            max_results=1
        )
        
        if not runs.empty:
            return runs.iloc[0].to_dict()
        return None
