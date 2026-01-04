#!/bin/bash

# Iniciar o Ray Head em background
# --dashboard-host=0.0.0.0 permite acesso externo ao dashboard
# --block faz o comando segurar o processo se fosse o único, mas queremos rodar ray start em background, 
# na verdade o ray start por padrão vai para background, mas para garantir que o container não morra,
# o ray start é um daemon.
ray start --head --dashboard-host=0.0.0.0 --port=6379 --num-cpus=4 --block &

# Aguarda o Ray iniciar
sleep 5

# Executa o comando passado para o container (padrão é bash no CMD do Dockerfile)
exec "$@"
