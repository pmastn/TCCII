import pandas as pd
from pymongo import MongoClient
from model_utils import preprocess_and_feature_engineer, train_model

# conectar mongodb
client = MongoClient("mongodb+srv://pmastnsantos125_db_user:TT7h3NVrOFbIcG8n@pvms.ai7o6wx.mongodb.net/?appName=pvms")
db = client["triagem_db"]
collection = db["atendimentos"]

# pegar dados corrigidos
registros = collection.find({"corrigido": True})

dados = []

for r in registros:
    entrada = dict(r["entrada"])

    # reconstruir formato do dataset original
    entrada["KTAS_expert"] = r["classe_real"]

    # garantir campo correto
    entrada["Chief_complain"] = entrada.get("Chief_complain_Grouped", "OUTROS")

    # garantir todos os campos numéricos
    entrada.setdefault("SBP", 120)
    entrada.setdefault("DBP", 80)
    entrada.setdefault("HR", 80)
    entrada.setdefault("RR", 18)
    entrada.setdefault("BT", 36.5)
    entrada.setdefault("Saturation", 98)
    entrada.setdefault("NRS_pain", 0)

    # idade obrigatória no pipeline
    entrada["Age"] = 30 
    dados.append(entrada)

# proteção
if len(dados) == 0:
    print("Nenhum dado corrigido encontrado.")
    exit()

df_novo = pd.DataFrame(dados)

print(f"Novos dados corrigidos: {len(df_novo)}")

# dataset original
df_original = pd.read_csv("data.csv", delimiter=';', encoding='latin1')

print(f"Dataset original: {len(df_original)}")

df_total = pd.concat([df_original, df_novo], ignore_index=True)

print(f"Dataset final: {len(df_total)}")

# salvar (debug e consistência)
df_total.to_csv("data_retrain.csv", sep=";", index=False)

# preprocessar
X, y = preprocess_and_feature_engineer("data_retrain.csv")

# treinar
train_model(X, y)

print("Modelo re-treinado com sucesso!")
