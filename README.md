tarefas:
- incluir o ray
- incluir o MLflow
- incluir corsmiddleware, o que é?
- incluir tags, onde? pra que?
- API
    - incluir informações no init para aparecer a versão app e autor
    - incluir o log handler
    - incluir os endpoints:
        - configurações
        - treinamento
        - inferência
        - atualização da rede: prunning, tuning, quantization
        - monitoramento
- docstrings
- Model deve ter:
    - init
    - forward
    - trainning_step
    - validation_step
    - configure_optimizer
- classe trainning strategy
- como aplicar factory no projeto?
- fazer sistemas de autenticação e autorização
- escrever o README com as decisões
- escrever o contribuiing.md
- escrever o security.md
- executar em um container docker
- como implantar?
- como configurar o logger?

yfinance - https://pypi.org/project/yfinance/





project/
│── Dockerfile
│── requirements.txt
│── main.py                # ponto de entrada da API
│── config.py               # configs globais (paths, env vars)
│
├── data/
│   └── raw/                # dados brutos baixados
│   └── processed/          # dados tratados
│
├── src/
│   ├── api/                # rotas da API (FastAPI/Flask)
│   │   ├── routes_data.py  # atualização de dados
│   │   ├── routes_model.py # predição, seleção, retreinamento
│   │   └── routes_utils.py # healthcheck, logs
│   │
│   ├── services/           # lógica de negócio
│   │   ├── data_loader.py  # módulo que chama yfinance
│   │   ├── model_manager.py# carrega, salva, troca modelos
│   │   └── predictor.py    # função de predição
│   │
│   ├── models/             # definição dos modelos ML
│   │   ├── arima.py
│   │   ├── lstm.py
│   │   └── baseline.py
│   │
│   └── utils/              # funções auxiliares (logs, métricas, etc.)
│
├── web/
│   ├── static/             # CSS, JS
│   ├── templates/          # HTML (se usar Jinja2)
│   └── app.py              # frontend simples (Flask/FastAPI + Jinja)
│
└── tests/                  # testes unitários e de integração