# Política de Segurança

Valorizamos muito a segurança deste projeto. Se você descobrir uma vulnerabilidade, pedimos que siga as diretrizes abaixo para nos ajudar a corrigi-la.

## Versões Suportadas

Atualmente, apenas a última versão na branch `main` recebe atualizações de segurança.

| Versão | Suportada          |
| ------- | ------------------ |
| v1.0.x | :white_check_mark: |
| < 1.0  | :x:                |

## Relatando uma Vulnerabilidade

**Por favor, não abra uma Issue pública para relatar vulnerabilidades de segurança.**

Envie um e-mail para o mantenedor do projeto ou utilize o sistema de relatórios privados do GitHub, caso disponível. Descreva detalhadamente:
1.  A natureza da vulnerabilidade.
2.  Como ela pode ser explorada.
3.  Possíveis correções imaginadas.

Prometemos analisar e responder ao seu relatório o mais rápido possível.

## Boas Práticas de Segurança no Projeto

Como este projeto lida com dados financeiros e APIs, recomendamos:
*   Não colocar senhas ou tokens de API diretamente no código. Use arquivos `.env`.
*   Sempre revisar as permissões de rede do Docker (portas expostas).
*   Garantir que o Dashboard do Ray e do Grafana estejam protegidos por firewall se forem expostos na internet.
