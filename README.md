# Qualidade do Sono

Repositório com código em Python relacionado ao tema **qualidade do sono**. O projeto contém scripts para tratamento de dados, visualização/gráficos, análise de correlação e rotinas de previsão/modelagem, além de um experimento inicial com Streamlit. (Baseado nos nomes dos arquivos presentes no repositório.)

> **Observação:** o repositório ainda não possui descrição oficial.  
> Fonte: página do repositório.  

## Estrutura do projeto

qualidadedosono/
├─ Dataset/ # Conjunto(s) de dados (pasta do repositório)
├─ correlacao.py # (Pelo nome) análise de correlação
├─ expondograficoslucas.py # (Pelo nome) geração/experimentos de gráficos
├─ grafico.py # (Pelo nome) utilitários para gráficos
├─ graficos.py # (Pelo nome) gráficos e visualizações
├─ graficosenzo.py # (Pelo nome) gráficos (variação/autor)
├─ tratamentododataset.py # (Pelo nome) limpeza e pré-processamento
├─ previsao.py # (Pelo nome) rotina de previsão
├─ modeloprevisao.py # (Pelo nome) modelo(s) de previsão
├─ modelo_previsao_final.py # (Pelo nome) versão final do modelo de previsão
├─ main.py # (Pelo nome) ponto de entrada
├─ main_final.py # (Pelo nome) ponto de entrada “final”
└─ steamlit_teste.py # (Pelo nome) teste com Streamlit

markdown
Copiar código

## Pré-requisitos

- **Python 3.x**
- As dependências específicas não estão listadas no repositório. Verifique os `import`s nos scripts que você pretende executar e instale os pacotes correspondentes manualmente (ex.: `pip install <pacote>`).

## Como usar

1. **Clone** este repositório:
   ```bash
   git clone https://github.com/lucashernandsz/qualidadedosono.git
   cd qualidadedosono
(Opcional) Crie um ambiente virtual:

bash
Copiar código
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
Instale as bibliotecas necessárias verificando os imports dos scripts que você vai rodar e executando pip install.

# Versão “final” sugerida pelos nomes dos arquivos
python main_final.py

# (Alternativas)
python main.py
python modelo_previsao_final.py
Dados: coloque os arquivos necessários dentro da pasta Dataset/ (já existente no repositório). Ajuste caminhos nos scripts se necessário.

Notas
Descrição, releases e tópicos não foram preenchidos na página do repositório até o momento.

Licença: não há licença explicitamente informada. Caso pretenda reutilizar o código, confirme com os autores e/ou adicione um arquivo LICENSE.

Contribuindo
Sinta-se à vontade para propor melhorias:

Abra uma issue descrevendo a sugestão/bug.

Envie um pull request com a alteração proposta.

Autores
Consulte a aba de contribuidores do GitHub para ver quem já participou do projeto.
