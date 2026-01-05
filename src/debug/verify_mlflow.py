
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.api.routes_model import router
from src.schemas.train_request import TrainRequest
from src.services.mlflow_service import MLFlowManager
from src.services.optimization_service import OptimizationService

from fastapi import FastAPI
app = FastAPI()
app.include_router(router)

class TestMLFeatures(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.api.routes_model.MLFlowManager")
    def test_list_models(self, mock_manager):
        # Setup mock
        instance = mock_manager.return_value
        instance.list_runs.return_value = [{"run_id": "123", "status": "FINISHED"}]
        
        response = self.client.get("/models/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"], [{"run_id": "123", "status": "FINISHED"}])

    @patch("src.api.routes_model.train_job")
    @patch("src.api.routes_model.TaskRegistry")
    def test_train_endpoint(self, mock_registry, mock_train_job):
        # Mock Ray actor
        mock_actor = MagicMock()
        mock_registry.remote.return_value = mock_actor
        
        request_data = {
            "model_type": "lstm",
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "window_size": 60,
            "log_level": "INFO"
        }
        
        response = self.client.post("/models/train", json=request_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("task_id", response.json()["data"])

    @patch("src.api.routes_model.tune_model")
    def test_tune_endpoint(self, mock_tune):
        request_data = {
            "model_type": "lstm",
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "window_size": 60,
            "log_level": "INFO"
        }
        response = self.client.post("/models/tune", json=request_data)
        self.assertEqual(response.status_code, 200)
        # Verify tune_model was called (it's a background task, so passing it to generic BackgroundTasks)
        # FastAPIs TestClient doesn't easily wait for background tasks, but we check if request was accepted
    
    @patch("src.api.routes_model.OptimizationService")
    def test_prune_endpoint(self, mock_service):
        instance = mock_service.return_value
        instance.prune_model.return_value = "Pruning completed"
        
        response = self.client.post("/models/run_123/prune?amount=0.3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Pruning completed")
        instance.prune_model.assert_called_with("run_123", 0.3)

    @patch("src.api.routes_model.MLFlowManager")
    def test_model_lifecycle(self, mock_manager):
        # Test Load
        instance = mock_manager.return_value
        mock_model = MagicMock()
        # Mock model call for prediction
        mock_model.return_value = MagicMock(tolist=lambda: [[0.1, 0.2]]) 
        instance.load_model.return_value = mock_model
        
        # Load
        response_load = self.client.post("/models/run_123/load")
        self.assertEqual(response_load.status_code, 200)
        
        # Predict
        response_predict = self.client.post("/models/predict", json={
            "model_run_id": "run_123",
            "data": [0.0] * 64 # Dummy data
        })
        self.assertEqual(response_predict.status_code, 200)
        self.assertEqual(response_predict.json()["data"], [[0.1, 0.2]])

if __name__ == '__main__':
    unittest.main()
