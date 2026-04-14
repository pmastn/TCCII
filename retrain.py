import pandas as pd
import sys
from pymongo import MongoClient
from model_utils import preprocess_and_feature_engineer, train_model


# CONEXÃO MONGODB

client = MongoClient("mongodb+srv://pmastnsantos125_db_user:TT7h3NVrOFbIcG8n@pvms.ai7o6wx.mongodb.net/?appName=pvms")

db = client["triagem_db"]
collection = db["atendimentos"]


# BUSCAR DADOS CORRIGIDOS

registros = list(collection.find({
    "corrigido": True,
    "classe_real": {"$ne": None}
}))

dados_novos = []

for r in registros:

    if "entrada" not in r:
        continue

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
    print("Nenhum dado corrigido.")
    exit()

df_novo = pd.DataFrame(dados_novos)

df_original = pd.read_csv("data.csv", delimiter=';', encoding='latin1')

# PESO SUAVE
total_novos = len(df_novo)
peso_novo = total_novos / (total_novos + 10)

df_original["peso"] = 1.0
df_novo["peso"] = peso_novo

df_total = pd.concat([df_original, df_novo], ignore_index=True)

print(f"Peso novos dados: {peso_novo:.3f}")

df_total.to_csv("data_retrain.csv", sep=";", index=False)

X, y = preprocess_and_feature_engineer("data_retrain.csv")

pesos = df_total["peso"].values

train_model(X, y, sample_weights=pesos)

print("Modelo re-treinado com sucesso!")
