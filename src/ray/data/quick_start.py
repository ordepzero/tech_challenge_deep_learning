import ray
import pandas as pd

class ClassificationModel:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", device=0)

    def __call__(self, batch: pd.DataFrame):
        results = self.pipe(list(batch["text"]))
        result_df = pd.DataFrame(results)
        return pd.concat([batch, result_df], axis=1)

ds = ray.data.read_text("s3://anonymous@ray-example-data/sms_spam_collection_subset.txt")
ds = ds.map_batches(
    ClassificationModel,
    compute=ray.data.ActorPoolStrategy(size=1),
    batch_size=64,
    batch_format="pandas",
    num_gpus=1  # this will set 1 GPU per worker
)
ds.show(limit=1)