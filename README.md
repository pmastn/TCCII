# Sistema de Triagem Hospitalar com IA

Projeto de TCC focado em apoio à decisão clínica na triagem hospitalar.

## Funcionalidades

- Classificação automática de risco (1 a 5)
- Correção médica dos casos
- Re-treinamento contínuo do modelo
- Dashboard com evolução do modelo

## Tecnologias

- FastAPI
- Streamlit
- XGBoost
- MongoDB

## Como rodar

1. Instalar dependências:
pip install -r requirements.txt

2. Rodar API:
uvicorn api:app --reload

3. Rodar interface:
streamlit run streamlit.py

## Observação

Se necessário, configurar string de conexão do MongoDB no arquivo api_fastapi.py