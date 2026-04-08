import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

# CONFIG

TARGET_COLUMN = 'KTAS_expert'

VITAL_SIGNS_AND_NUMERIC = [
    'SBP','DBP','HR','RR','BT','Saturation','NRS_pain'
]

CATEGORICAL_FEATURES = [
    'Sex','Chief_complain_Grouped','Age_Group'
]

LOW_FREQUENCY_THRESHOLD = 10


# FUNÇÃO AUX
def clean_and_convert_to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')


# PRÉ-PROCESSAMENTO (FUNCIONA COM DF OU CSV) 
def preprocess_and_feature_engineer(data):

    # aceita string (arquivo) OU DataFrame
    if isinstance(data, str):
        df = pd.read_csv(data, delimiter=';', encoding='latin1')
    else:
        df = data.copy()

    # colunas iniciais
    all_columns_initial = [TARGET_COLUMN,'Age'] + VITAL_SIGNS_AND_NUMERIC + ['Sex','Chief_complain']
    df_model = df[all_columns_initial].copy()

    # PROCESSAMENTO DE TEXTO
    df_model['Chief_complain'] = df_model['Chief_complain'].str.lower()

    complaint_mapping = {
        'abd pain':'abdo pain',
        'abd. pain':'abdo pain',
        'abdomen pain':'abdo pain'
    }

    df_model['Chief_complain'].replace(complaint_mapping, inplace=True)

    complaint_counts = df_model['Chief_complain'].value_counts()

    rare_complaints = complaint_counts[complaint_counts < LOW_FREQUENCY_THRESHOLD].index

    df_model['Chief_complain_Grouped'] = df_model['Chief_complain'].apply(
        lambda x: 'OUTROS' if x in rare_complaints else x
    )

    df_model.drop(columns=['Chief_complain'], inplace=True)

    # NUMÉRICOS 
    df_model['Age'] = clean_and_convert_to_numeric(df_model['Age'])

    for col in VITAL_SIGNS_AND_NUMERIC:
        df_model[col] = clean_and_convert_to_numeric(df_model[col])

    df_model['Age'].fillna(df_model['Age'].median(), inplace=True)

    for col in VITAL_SIGNS_AND_NUMERIC:
        df_model[col].fillna(df_model[col].median(), inplace=True)

    #IDADE EM GRUPOS
    df_model['Age_Group'] = pd.cut(
        df_model['Age'],
        bins=[0,2,12,17,59,120],
        labels=['Bebe','Crianca','Adolescente','Adulto','Idoso']
    )

    df_model.drop(columns=['Age'], inplace=True)

    df_model.dropna(inplace=True)

    # ONE HOT ENCODING 
    df_encoded = pd.get_dummies(
        df_model,
        columns=CATEGORICAL_FEATURES,
        drop_first=True
    )

    X = df_encoded.drop(columns=[TARGET_COLUMN])
    y = df_encoded[TARGET_COLUMN]

    return X, y


# TREINAMENTO SIMPLES (USADO NA API)
def train_model(X, y):

    y_adjusted = y - 1  # classes de 0 a 4

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # balanceamento de classes 
    classes = np.unique(y_adjusted)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_adjusted
    )

    class_weights = dict(zip(classes, weights))

    sample_weights = np.array([
        class_weights[label] for label in y_adjusted
    ])

    model.fit(X, y_adjusted, sample_weight=sample_weights)

    # salvar modelo e features
    joblib.dump(model, "modelo_xgboost.pkl")
    joblib.dump(X.columns.tolist(), "features_modelo.pkl")

    return model
