import ray
import logging

# Configuração básica de log para saída padrão (stdout)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_registry")

@ray.remote
class TaskRegistry:
    """
    Ator remoto do Ray que atua como um registro centralizado para o status de tarefas em segundo plano.
    """
    def __init__(self):
        """
        Inicializa o dicionário de tarefas.
        """
        self.tasks = {}
        logger.info("TaskRegistry inicializado.")

    def set(self, task_id, state):
        """
        Define ou atualiza o estado de uma tarefa específica.
        """
        logger.info(f"SET task_id={task_id} state={state}")
        self.tasks[task_id] = {"state": state}
        # Print para garantir visibilidade nos logs do Docker
        print(f"[REGISTRY-DEBUG] SET {task_id} -> {state}")
        return True

    def get(self, task_id):
        """
        Recupera o estado de uma tarefa através do seu ID.
        """
        val = self.tasks.get(task_id)
        logger.info(f"GET task_id={task_id} retornando={val}")
        print(f"[REGISTRY-DEBUG] GET {task_id} -> {val}")
        return val

    def list(self):
        """
        Lista todas as tarefas registradas e seus respectivos estados.
        """
        logger.info(f"LIST retornando {len(self.tasks)} itens")
        print(f"[REGISTRY-DEBUG] LIST {len(self.tasks)} itens")
        return self.tasks
