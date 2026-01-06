
import mlflow.pytorch
import torch
import torch.nn.utils.prune as prune
import os
import sys

# Ensure tracking URI is set correctly as in the app
os.environ["MLFLOW_TRACKING_URI"] = "file:///app/mlruns"
# Assuming /app is PYTHONPATH or similar, though imports usually work if in /app
sys.path.append("/app")

run_id = "55d56fcdf1004ff09720984506a24ea6"
model_uri = f"runs:/{run_id}/model"

print(f"Attempting to load model from {model_uri}...")

try:
    model = mlflow.pytorch.load_model(model_uri)
    print("Model loaded successfully!")
    print(model)
    
    print("Attempting to prune...")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Pruning layer: {name}")
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight')
            count += 1
            
    print(f"Pruning complete. {count} layers pruned.")
    
except Exception as e:
    print("FAILED with error:")
    import traceback
    traceback.print_exc()
