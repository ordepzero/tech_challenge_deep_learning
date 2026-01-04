import ray

@ray.remote
class TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def set(self, task_id, state):
        self.tasks[task_id] = {"state": state}

    def get(self, task_id):
        return self.tasks.get(task_id)

    def list(self):
        return self.tasks