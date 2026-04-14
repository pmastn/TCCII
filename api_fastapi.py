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

    collection.insert_one(registro)

    return {
        "classe_predita": int(classe),
        "probabilidades": probabilidades.tolist()
    }



# CORREÇÃO

@app.post("/corrigir")
def corrigir(dados: dict):

    cpf = dados.get("cpf")
    classe_real = dados.get("classe_real")

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

    return {"status": "ok"}



# RETRAIN
@app.post("/retrain")
def retrain():

    global modelo, features_modelo, modelo_versao

    try:
        from model_utils import preprocess_and_feature_engineer, train_model

        registros = list(collection.find({
            "corrigido": True,
            "classe_real": {"$ne": None}
        }))

        dados_novos = []

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

            dados_novos.append(entrada)

        if len(dados_novos) == 0:
            return {"erro": "Sem dados corrigidos"}

        df_novo = pd.DataFrame(dados_novos)

        df_original = pd.read_csv("data.csv", delimiter=';', encoding='latin1')

        # PESO SUAVE
        total_novos = len(df_novo)
        peso_novo = total_novos / (total_novos + 10)

        df_original["peso"] = 1.0
        df_novo["peso"] = peso_novo

        df_total = pd.concat([df_original, df_novo], ignore_index=True)

        df_total.to_csv("data_retrain.csv", sep=";", index=False)

        X, y = preprocess_and_feature_engineer("data_retrain.csv")

        pesos = df_total["peso"].values

        modelo = train_model(X, y, sample_weights=pesos)

        features_modelo = joblib.load("features_modelo.pkl")

        modelo_versao += 1

        return {
            "status": "modelo atualizado",
            "versao": modelo_versao,
            "peso_novos": round(peso_novo, 3)
        }

    except Exception as e:
        return {"erro": str(e)}



# DADOS
@app.get("/dados")
def dados():
    return list(collection.find({}, {"_id": 0}))
