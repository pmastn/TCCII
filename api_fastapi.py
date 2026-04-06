from fastapi import FastAPI
import pandas as pd
import joblib
from pymongo import MongoClient
from datetime import datetime

app = FastAPI()

modelo_versao = 1

# CONEXÃO COM MONGODB

client = MongoClient("mongodb+srv://pmastnsantos125_db_user:TT7h3NVrOFbIcG8n@pvms.ai7o6wx.mongodb.net/?appName=pvms")

db = client["triagem_db"]
collection = db["atendimentos"]


# CARREGAR MODELO

modelo = joblib.load("modelo_xgboost.pkl")
features_modelo = joblib.load("features_modelo.pkl")

#
# ROTA INICIAL

@app.get("/")
def home():
    return {"mensagem": "API de triagem hospitalar ativa"}



# PREDIÇÃO

@app.post("/predict")
def predict(dados: dict):

    cpf = dados.get("cpf")

    if not cpf:
        return {"erro": "CPF é obrigatório"}

    dados_modelo = dados.copy()
    dados_modelo.pop("cpf", None)

    # garantir campos obrigatórios
    dados_modelo.setdefault("Chief_complain_Grouped", "OUTROS")

    df = pd.DataFrame([dados_modelo])
    df = pd.get_dummies(df)

    for col in features_modelo:
        if col not in df.columns:
            df[col] = 0

    df = df[features_modelo]

    probabilidades = modelo.predict_proba(df)[0]
    classe = probabilidades.argmax() + 1

    registro = {
        "cpf": cpf,
        "entrada": dados_modelo,
        "classe_predita": int(classe),
        "classe_real": None,
        "corrigido": False,
        "probabilidades": probabilidades.tolist(),
        "timestamp_predicao": datetime.utcnow(),
        "modelo_versao": modelo_versao  
    }

    try:
        collection.insert_one(registro)
    except Exception as e:
        print("Erro ao salvar no Mongo:", e)

    return {
        "classe_predita": int(classe),
        "probabilidades": probabilidades.tolist()
    }



# CORREÇÃO MÉDICA

@app.post("/corrigir")
def corrigir(dados: dict):

    cpf = dados.get("cpf")
    classe_real = dados.get("classe_real")

    if not cpf:
        return {"erro": "CPF é obrigatório"}

    if classe_real is None:
        return {"erro": "classe_real é obrigatória"}

    registro = collection.find_one(
        {"cpf": cpf},
        sort=[("timestamp_predicao", -1)]
    )

    if not registro:
        return {"erro": "CPF não encontrado"}

    collection.update_one(
        {"_id": registro["_id"]},
        {
            "$set": {
                "classe_real": classe_real,
                "corrigido": True,
                "timestamp_correcao": datetime.utcnow()
            }
        }
    )

    return {"status": "correção salva com sucesso"}



# RETREINAMENTO

@app.post("/retrain")
def retrain():

    global modelo, features_modelo, modelo_versao  # 🔥 IMPORTANTE

    try:
        from model_utils import preprocess_and_feature_engineer, train_model

        registros = collection.find({"corrigido": True})

        dados = []

        for r in registros:
            entrada = dict(r["entrada"])

            entrada["KTAS_expert"] = r["classe_real"]
            entrada["Chief_complain"] = entrada.get("Chief_complain_Grouped", "OUTROS")

            entrada.setdefault("SBP", 120)
            entrada.setdefault("DBP", 80)
            entrada.setdefault("HR", 80)
            entrada.setdefault("RR", 18)
            entrada.setdefault("BT", 36.5)
            entrada.setdefault("Saturation", 98)
            entrada.setdefault("NRS_pain", 0)

            entrada["Age"] = 30

            dados.append(entrada)

        if len(dados) == 0:
            return {"erro": "Nenhum dado corrigido para treinar"}

        df_novo = pd.DataFrame(dados)

        df_original = pd.read_csv("data.csv", delimiter=';', encoding='latin1')

        df_total = pd.concat([df_original, df_novo], ignore_index=True)

        print(f"Re-treinando com {len(df_total)} registros...")

        df_total.to_csv("data_retrain.csv", sep=";", index=False)

        X, y = preprocess_and_feature_engineer("data_retrain.csv")

        # 🧠 TREINA NOVO MODELO
        modelo = train_model(X, y)

        # 🔄 ATUALIZA FEATURES
        features_modelo = joblib.load("features_modelo.pkl")

        # 🔥 ESSA LINHA É A MAIS IMPORTANTE DE TODAS
        modelo_versao += 1

        print(f"Modelo atualizado! Nova versão: {modelo_versao}")

        return {
            "status": "modelo re-treinado com sucesso",
            "nova_versao": modelo_versao
        }

    except Exception as e:
        return {"erro": str(e)}
    


@app.get("/dados")
def get_dados():

    registros = list(collection.find({}, {"_id": 0}))

    return registros