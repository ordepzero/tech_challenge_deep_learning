import ray
import logging

# Configure basic logging to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_registry")

@ray.remote
class TaskRegistry:
    def __init__(self):
        self.tasks = {}
        logger.info("TaskRegistry initialized.")

    def set(self, task_id, state):
        logger.info(f"SET task_id={task_id} state={state}")
        self.tasks[task_id] = {"state": state}
        # Print to stdout/stderr so it appears in docker logs
        print(f"[REGISTRY-DEBUG] SET {task_id} -> {state}")
        return True

    def get(self, task_id):
        val = self.tasks.get(task_id)
        logger.info(f"GET task_id={task_id} returning={val}")
        print(f"[REGISTRY-DEBUG] GET {task_id} -> {val}")
        return val

    def list(self):
        logger.info(f"LIST returning {len(self.tasks)} items")
        print(f"[REGISTRY-DEBUG] LIST {len(self.tasks)} items")
        return self.tasks