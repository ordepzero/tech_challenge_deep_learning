# Guia de Contribuição

Ficamos muito felizes que você tenha interesse em contribuir para este projeto! Como um projeto focado em Deep Learning e Engenharia de Dados, seguimos algumas diretrizes para manter o código organizado e funcional.

## Como Contribuir

### 1. Relatando Bugs
Se você encontrar um erro (bug), abra uma **Issue** detalhando:
*   O comportamento esperado e o comportamento observado.
*   Passos para reproduzir o erro.
*   Seu ambiente (versão do Docker, se está usando GPU, etc.).

### 2. Sugerindo Melhorias
Toda ideia é bem-vinda! Abra uma Issue com a tag `enhancement` descrevendo a funcionalidade e por que ela seria útil para o projeto.

### 3. Enviando Pull Requests (PRs)
Se você quer colocar a mão na massa:
1.  Faça um **Fork** do repositório.
2.  Crie uma branch para sua modificação (`git checkout -b feature/minha-melhoria`).
3.  Garanta que seu código segue o estilo do projeto:
    *   Use docstrings em português.
    *   Mantenha nomes de variáveis descritivos.
    *   Se estiver adicionando um novo modelo, utilize a `ModelFactory`.
4.  Certifique-se de que os containers sobem corretamente com suas mudanças.
5.  Envie o PR detalhando o que foi alterado.

## Padrões de Código

*   **Linguagem**: Documentação e comentários devem ser em **Português**. O código (nomes de funções e variáveis) segue a convenção padrão em **Inglês**.
*   **Frameworks**: Priorize o uso de `PyTorch Lightning` para novos modelos.
*   **Async**: Como usamos FastAPI, prefira endpoints assíncronos quando lidar com E/S se possível.

## Licença
Ao contribuir, você concorda que seu código será distribuído sob a mesma licença do projeto (MIT ou similar).
