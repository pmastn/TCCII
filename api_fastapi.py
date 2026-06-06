from fastapi import FastAPI
import pandas as pd
import joblib
from pymongo import MongoClient
from datetime import datetime
import math
from model_utils import agrupar_queixa, TODOS_SINTOMAS

app = FastAPI()

modelo_versao = 1

# CONEXÃO COM MONGODB
client = MongoClient(
    "mongodb+srv://pmastnsantos125_db_user:TT7h3NVrOFbIcG8n@pvms.ai7o6wx.mongodb.net/?appName=pvms"
)

db = client["triagem_db"]
sintomas_collection = db["sintomas_catalogo"]
collection = db["atendimentos"]


# CARREGAR MODELO
modelo = joblib.load("modelo_xgboost.pkl")
features_modelo = joblib.load("features_modelo.pkl")


# ROTA INICIAL
@app.get("/")
def home():
    return {"mensagem": "API de triagem hospitalar ativa"}


# =========================
# PREDIÇÃO
# =========================
@app.post("/predict")
def predict(dados: dict):

    cpf = dados.get("cpf")

    if not cpf:
        return {"erro": "CPF é obrigatório"}

    dados_modelo = dados.copy()
    dados_modelo.pop("cpf", None)

    queixa_original = dados_modelo.get(
        "Chief_complain",
        dados_modelo.get("Chief_complain_Grouped", "OUTROS")
    ).lower().strip()

    dados_modelo.pop("Chief_complain_Grouped", None)

    sintoma_catalogado = sintomas_collection.find_one({
        "sintoma": queixa_original,
        "aprovado": True
    })

    sintoma_desconhecido = (
        queixa_original not in TODOS_SINTOMAS
        and sintoma_catalogado is None
    )

    if sintoma_catalogado:
        grupo_semantico = sintoma_catalogado.get(
            "grupo_semantico",
            agrupar_queixa(queixa_original)
        )
    else:
        grupo_semantico = agrupar_queixa(queixa_original)

    dados_modelo["Chief_complain"] = queixa_original
    dados_modelo["Semantic_Group"] = grupo_semantico

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
        "modelo_versao": modelo_versao,
        "grupo_semantico": grupo_semantico,
        "sintoma_desconhecido": sintoma_desconhecido
    }

    collection.insert_one(registro)

    return {
        "classe_predita": int(classe),
        "probabilidades": probabilidades.tolist(),
        "sintoma_desconhecido": sintoma_desconhecido,
        "grupo_semantico": grupo_semantico
    }

# =========================
# CORREÇÃO
# =========================
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


@app.post("/corrigir_grupo")
def corrigir_grupo(dados: dict):

    cpf = dados.get("cpf")
    novo_grupo = dados.get("grupo_semantico")

    if not cpf or not novo_grupo:
        return {"erro": "CPF e grupo semântico são obrigatórios"}

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
                "entrada.Semantic_Group": novo_grupo,
                "grupo_semantico": novo_grupo,
                "grupo_corrigido": True,
                "sintoma_desconhecido": False
            }
        }
    )

    sintoma = registro["entrada"].get(
        "Chief_complain",
        ""
    ).lower().strip()

    if sintoma:
        sintomas_collection.update_one(
            {"sintoma": sintoma},
            {
                "$set": {
                    "sintoma": sintoma,
                    "grupo_semantico": novo_grupo,
                    "aprovado": True,
                    "timestamp_aprovacao": datetime.utcnow()
                }
            },
            upsert=True
        )

    return {
        "status": "grupo atualizado",
        "sintoma_aprovado": sintoma
    }


@app.get("/sintomas")
def listar_sintomas():
    sintomas_base = list(TODOS_SINTOMAS)

    sintomas_extra = [
        s["sintoma"]
        for s in sintomas_collection.find(
            {"aprovado": True},
            {"_id": 0, "sintoma": 1}
        )
    ]

    return sorted(set(sintomas_base + sintomas_extra))

# =========================
# RETRAIN
# =========================
@app.post("/retrain")
def retrain():

    global modelo, features_modelo, modelo_versao

    try:
        from model_utils import (
            preprocess_and_feature_engineer,
            train_model
        )

        registros = list(collection.find({
            "corrigido": True,
            "classe_real": {"$ne": None},
            "usado_no_retreino": {"$ne": True}
        }))

        dados_novos = []

        idade_map = {
            "Bebe": 1,
            "Crianca": 8,
            "Adolescente": 15,
            "Adulto": 30,
            "Idoso": 70
        }

        for r in registros:
            entrada = dict(r["entrada"])

            entrada["KTAS_expert"] = r["classe_real"]

            entrada["Chief_complain"] = entrada.get(
                "Chief_complain",
                "outros clinicos"
            ).lower().strip()

            entrada["Semantic_Group"] = entrada.get(
                "Semantic_Group",
                r.get("grupo_semantico", agrupar_queixa(entrada["Chief_complain"]))
            )

            entrada.setdefault("SBP", 120)
            entrada.setdefault("DBP", 80)
            entrada.setdefault("HR", 80)
            entrada.setdefault("RR", 18)
            entrada.setdefault("BT", 36.5)
            entrada.setdefault("Saturation", 98)
            entrada.setdefault("NRS_pain", 0)

            entrada["SBP"] = float(entrada["SBP"])
            entrada["DBP"] = float(entrada["DBP"])
            entrada["HR"] = float(entrada["HR"])
            entrada["RR"] = float(entrada["RR"])
            entrada["BT"] = float(entrada["BT"])
            entrada["Saturation"] = float(entrada["Saturation"])
            entrada["NRS_pain"] = float(entrada["NRS_pain"])


            entrada["Age"] = idade_map.get(
                entrada.get("Age_Group", "Adulto"),
                30
            )

            dados_novos.append(entrada)

        if len(dados_novos) == 0:
            return {"erro": "Sem dados corrigidos"}

        MINIMO_RETRAIN = 10

        if len(dados_novos) < MINIMO_RETRAIN:
            return {
                "erro": f"Re-treinamento bloqueado: são necessários pelo menos {MINIMO_RETRAIN} novos casos confirmados."
            }

        df_novo = pd.DataFrame(dados_novos)

        df_original = pd.read_csv(
            "data.csv",
            delimiter=';',
            encoding='latin1'
        )

        total_original = len(df_original)
        total_novos = len(df_novo)

        # PESO DINÂMICO 
        peso_novo = 1 + (
            math.log10(total_original / max(total_novos, 1)) * 0.08
        )

        # limites de segurança
        peso_novo = max(1.05, min(peso_novo, 1.4))

        peso_novo = round(peso_novo, 2)

        df_original["peso"] = 1.0
        df_novo["peso"] = peso_novo

        df_total = pd.concat(
            [df_original, df_novo],
            ignore_index=True
        )

        df_total.to_csv(
            "data_retrain.csv",
            sep=";",
            index=False
        )

        X, y, indices_validos = preprocess_and_feature_engineer(
            "data_retrain.csv"
        )

        df_total["peso"] = df_total["peso"].fillna(1.0)
        pesos = df_total.loc[indices_validos, "peso"].values

        modelo = train_model(
            X,
            y,
            sample_weights=pesos
        )

        features_modelo = joblib.load(
            "features_modelo.pkl"
        )

        modelo_versao += 1

        collection.update_many(
            {
                "corrigido": True,
                "classe_real": {"$ne": None},
                "usado_no_retreino": {"$ne": True}
            },
            {
                "$set": {
                    "usado_no_retreino": True
                }
            }
        )

        return {
            "status": "modelo atualizado",
            "versao": modelo_versao,
            "peso_novos": round(peso_novo, 3)
        }

    except Exception as e:
        return {"erro": str(e)}


# =========================
# DADOS
# =========================
@app.get("/dados")
def dados():
    return list(
        collection.find({}, {"_id": 0})
    )
