import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

TARGET_COLUMN = 'KTAS_expert'

VITAL_SIGNS_AND_NUMERIC = [
    'SBP','DBP','HR','RR','BT','Saturation','NRS_pain'
]

CATEGORICAL_FEATURES = [
    'Sex','Chief_complain_Grouped','Age_Group'
]

LOW_FREQUENCY_THRESHOLD = 10


def clean_and_convert_to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')


def preprocess_and_feature_engineer(file_path):

    df = pd.read_csv(file_path, delimiter=';', encoding='latin1')

    all_columns_initial = [TARGET_COLUMN,'Age'] + VITAL_SIGNS_AND_NUMERIC + ['Sex','Chief_complain']
    df_model = df[all_columns_initial].copy()

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

    # numéricos
    df_model['Age'] = clean_and_convert_to_numeric(df_model['Age'])

    for col in VITAL_SIGNS_AND_NUMERIC:
        df_model[col] = clean_and_convert_to_numeric(df_model[col])

    df_model['Age'].fillna(df_model['Age'].median(), inplace=True)

    for col in VITAL_SIGNS_AND_NUMERIC:
        df_model[col].fillna(df_model[col].median(), inplace=True)

    # idade categórica
    df_model['Age_Group'] = pd.cut(
        df_model['Age'],
        bins=[0,2,12,17,59,120],
        labels=['Bebe','Crianca','Adolescente','Adulto','Idoso']
    )

    df_model.drop(columns=['Age'], inplace=True)

    df_model.dropna(inplace=True)

    # one-hot
    df_encoded = pd.get_dummies(
        df_model,
        columns=CATEGORICAL_FEATURES,
        drop_first=True
    )

    X = df_encoded.drop(columns=[TARGET_COLUMN])
    y = df_encoded[TARGET_COLUMN]

    return X, y


def train_model(X, y, sample_weights=None):

    y_adjusted = y - 1

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

    model.fit(X, y_adjusted, sample_weight=sample_weights)

    joblib.dump(model, "modelo_xgboost.pkl")
    joblib.dump(X.columns.tolist(), "features_modelo.pkl")

    return model
