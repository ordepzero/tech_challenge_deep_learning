https://www.youtube.com/watch?v=CbTU92pbDKw

passos:
- importar pandas, numpy, matplotlib.pyplot
- extrair dados históricos de um papel da bolsa
- possui dados de abertura, fechamento, máximo, mínimo e volume de negociação, data
- converter o campo de data para o tipo date
- converter o campo de data em index
- plot dados de fechamento por dia
- criar uma função de janela recebe o dataframe e é baseada na data inicial, data final, n =3 (o que faz essa função?)
- essa função cria para cada data (target date) no dataframe ele busca o valor de fechamento de "n" datas anteriores
- as datas que possuem valores zerados não são consideradas no dataframe final
- criar função que converte o dataframe e separa em 3 arrays numpy: dates, X, y
- separa as bases em treino, validação e teste: primeiros 80% de treino, de 80% a 90% de validação, 90% a 100% de teste
- a seperação considera a ordem cronológica

modelagem:
- criação de uma sequência: input, LSTM, dense 32 ativação relu, dense 32 ativação relu, dense 1
- compile loss mse, optimizer Adam, metrics mean absolute error 
- executa o fit
- plotar o gráfico das previsões de treino, validação e teste, separadamente    
- se resultado não for bom, diminuir o intervalo de dados de treinamento
-- será se não é necessário normalizar os dados?
-- podemos treinar com séries maiores?
-- qual a melhor granualidade de dados?





funções utilizadas:
- reshape
- squeeze

- é possível habilidatar o log dos hiper parâmetros
- aplicar quantização
- o que faz a função loss?
- o que é o mse puro?

- mudar target para retornos logaritmos
- tentar prever o log return
- como usar: a rede prevê o retorno, e reconstroi o preço final multiplicando
o último valor conhecido pelo exponencial da previsão
- tentar o MAPE
- tentar um custom loss: criar uma função de perda que penaliza especificamente o erro de direção
( se a rede previu alta e foi baixa, a penalidade é dobrada)
- aplicar a diferencição, assim rede aprende a mudança e não o patamar de preço
- normalizar com o primeiro valor da série

- problema do lag effect
-- usar indicadores técnicos: RSI, MACD, médias móveis, volume
-- usar retorno logaritmo
-- usar o volume? como usar um valor que será tão alto em um rede neural?

reduzir a complexidade da rede
incluir dropout




Ray
- Como saber quantos workers tem minha máquina?
    - você pode verificar o valor da variável num_workers no seu código
    - você também pode verificar o número total de CPUs (ray.cluster_resources()["CPU"])



python -c "import pyarrow; print(pyarrow.__version__)"


### Docker:

- Montar a imagem que está no Dockerfile:
$ docker build -t ray-ml-py312-gpu .
$ docker build -t ray-ml-nightly-py312-gpu .
$ docker build -t ray-ml-2.31.0-py310-gpu .
$ docker build -t pytorchlightning-py312-gpu .
$ docker build -t stocks .   

- Rodar com GPU e memória compartilhada maior
$ docker run --gpus all -it --shm-size=16g -p 8265:8265 -v ${PWD}:/app ray-ml-py312-gpu
$ docker run --shm-size=16g -t -i --gpus all -p 8265:8265 -v ${PWD}:/app ray-ml-py312-gpu
$ docker run --shm-size=16g -t -i --gpus all -p 8265:8265 -v ${PWD}:/app ray-ml-nightly-py312-gpu
$ docker run --shm-size=16g -t -i --gpus all -p 8265:8265 -v ${PWD}:/app ray-ml-nightly-extra-py310
$ docker run --shm-size=16g -t -i --gpus all -p 8265:8265 -v ${PWD}:/app ray-ml-2.31.0-py310-gpu
$ docker run --shm-size=20g -t -i --gpus all -p 8265:8265 -v ${PWD}:/app pytorchlightning-py312-gpu
$ docker run --shm-size=20g -t -i --gpus all -p 8265:8265 -v ${PWD}:/app bitnami
$ docker run --shm-size=20g --gpus all -p 8265:8265 -p 8000:8000 -v ${PWD}:/app -it stocks

- Testar PyTorch + GPU dentro do contêiner
$ python -c "import torch; import lightning; print(torch.__version__, torch.cuda.is_available(), lightning.__version__)

$ python -m src.models.ray_tuning_2
$ python -m ray_lightning


pip show ray lightning torch


ray start --head --dashboard-host=0.0.0.0
ray stop
ray start --head --dashboard-host=0.0.0.0 --port=6379

docker ps
docker exec -it <CONTAINER_ID> python meu_modulo.py
docker exec -it <CONTAINER_ID> bash
docker exec -it e8a5c8312a3e bash

docker start -ai 7f5ff63d71b484c5a9043389056843ebf60d91c52b7ecd67cf0ae721ece003d9
docker exec -it 958283c0aed05418556cdc968337fc04ab95566b69595d35793a5c0edc4bdfb5 bash   