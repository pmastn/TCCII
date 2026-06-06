import pandas as pd
from xgboost import XGBClassifier
import joblib

TARGET_COLUMN = 'KTAS_expert'

VITAL_SIGNS_AND_NUMERIC = [
    'SBP', 'DBP', 'HR', 'RR', 'BT',
    'Saturation', 'NRS_pain'
]

CATEGORICAL_FEATURES = [
    'Sex',
    'Chief_complain',
    'Semantic_Group',
    'Age_Group'
]


def clean_and_convert_to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.'),
        errors='coerce'
    )


# ======================================
# AGRUPAMENTO SEMÂNTICO FINAL
# ======================================
def agrupar_queixa(queixa):

    q = str(queixa).lower().strip()

    # =========================
    # NORMALIZAÇÃO / SINÔNIMOS
    # =========================
    complaint_mapping = {

    # =====================
    # ABDOMINAL
    # =====================
    'abd pain': 'abdo pain',
    'abd. pain': 'abdo pain',
    'abdomen pain': 'abdo pain',
    'pain, abdomen': 'abdo pain',
    'pain, abdominal': 'abdo pain',
    'abdominal pain (finding)': 'abdo pain',

    'rt abd pain': 'right lower abdominal pain',
    'rlq abd. pain': 'right lower quadrant abdominal pain',
    'rlq abdominal pain': 'right lower quadrant abdominal pain',

    'low abd. pain': 'low abdominal pain',
    'low abd discomfort': 'low abdominal discomfort',

    'distension, abd': 'abd distension',
    'abdomen distension': 'abd distension',
    'abd. distension': 'abd distension',
    'distended abdomen': 'abd distension',

    'diffuse abdominal discomfort': 'abdominal discomfort',
    'abdomen discomfort': 'abdominal discomfort',

    # =====================
    # DIARREIA / VÔMITO
    # =====================
    'd - diarrhea': 'diarrhea',
    'watery diarrhea': 'diarrhea',

    'v - vomiting': 'vomiting',

    # =====================
    # FEBRE
    # =====================
    'fever & chill': 'fever',
    'f/c-fever/chills': 'fever',
    'fever, unspecified': 'fever',
    'f/c, c/s': 'fever',

    # =====================
    # HEADACHE
    # =====================
    'h-headache': 'headache',
    'ha': 'headache',
    'ha - headache': 'headache',
    'occipital area headache': 'headache',

    # =====================
    # DIZZINESS
    # =====================
    'dz - dizziness': 'dizziness',
    'rotatory vertigo': 'dizziness',
    'whirling type vertigo': 'dizziness',
    'light headness': 'dizziness',

    # =====================
    # MENTAL
    # =====================
    'altered mental change': 'mental change',
    'altered mentality': 'mental change',
    'mental confusion': 'mental change',
    'confuse mentality': 'mental change',
    'confused mental status': 'mental change',
    'confusion mentality': 'mental change',
    'confusion state': 'mental change',
    'behavior change': 'mental change',
    'acute delirium': 'mental change',
    'drowsy mentality': 'mental change',

    # =====================
    # HEMIPARESIS / WEAKNESS
    # =====================
    'right hemiparesis': 'rt hemiparesis',
    'lt. hemiparesis': 'lt hemiparesis',

    'rt. side motor weakness': 'right motor weakness',
    'rt.arm weakness': 'right motor weakness',

    'lt. motor weakness': 'left motor weakness',
    'lt. side weakness': 'left motor weakness',
    'l side weakness': 'left motor weakness',
    'left side weakness': 'left motor weakness',
    'left side motor weakness': 'left motor weakness',

    'rigth side weakness': 'right side weakness',

    # =====================
    # SEIZURE
    # =====================
    'post seizure': 'seizure',
    'seizure like activity': 'seizure',
    'sezure like motion': 'seizure',
    'convulsion': 'seizure',

    # =====================
    # LOC
    # =====================
    'loc': 'loss of consciousness',
    'loc - loss of consciousness': 'loss of consciousness',
    'near syncope': 'syncope',

    # =====================
    # CHEST PAIN
    # =====================
    'pain, chest': 'chest pain',
    'chest pain nos (finding)': 'chest pain',

    'right chest pain': 'rt. chest pain',
    'chest pain rt': 'rt. chest pain',

    'ant. chest pain': 'anterior chest pain',

    # =====================
    # CHEST DISCOMFORT
    # =====================
    'discomfort, chest': 'chest discomfort',
    'diffuse chest discomfort': 'chest discomfort',
    'with chest discomfort': 'chest discomfort',

    # =====================
    # COUGH
    # =====================
    'c - coughing': 'cough',

    # =====================
    # DYSPNEA
    # =====================
    'acute dyspnea': 'dyspnea',

    # =====================
    # HEMOPTYSIS
    # =====================
    'blood tinged sputum': 'hemoptysis',

    # =====================
    # EYE
    # =====================
    'eye pain left eye': 'left eye pain',
    'ocular pain, lt.': 'left eye pain',
    'ocular pain, rt.': 'right ocular pain',
    'ocular pain, both': 'both ocular pain',

    'fb in eye': 'eye foreign body',

    # =====================
    # LACERATION
    # =====================
    'finger lac.': 'finger laceration',
    'finger laceration wound': 'finger laceration',

    'laceration, scalp': 'scalp laceration',
    'scalp lac.': 'scalp laceration',

    # =====================
    # URINARY
    # =====================
    'difficulty in voiding': 'voiding difficulty',
    'urinary sx.-symptom': 'urinary symptom',

    # =====================
    # RASH
    # =====================
    'urticarial rash': 'skin rash',
    'whole body, skin rash': 'skin rash',

    # =====================
    # TOOTH
    # =====================
    'toothache': 'tooth pain',

    }

    q = complaint_mapping.get(q, q)

    # =========================
    # AGRUPAMENTOS SEMÂNTICOS
    # =========================

    neurologicos = {
        'headache',
        'dizziness',
        'mental change',
        'dysarthria',
        'syncope',
        'loss of consciousness',
        'amnesia',

        'rt hemiparesis',
        'lt hemiparesis',
        'right motor weakness',
        'left motor weakness',
        'right side weakness',
        'right monoparesis',
        'lower extremity paraparesis',

        'facial palsy',
        'facial palsy left',
        'left facial numbness',
        'aphasia, motor',

        'seizure',
        'blurred vision',
        'visual disturbance',
        'abnormality, visual acuity',
        'decerased visual acuity',
        'left diplopia',

        'limb paresthesia',
        'numbness',
        'rt side numbness',
        'both hand numbness',
        'hand tingling sense'
    }


    cardiacos = {
        'chest pain',
        'left chest pain',
        'rt. chest pain',
        'anterior chest pain',

        'chest discomfort',
        'chest discomfort left',
        'anterior chest discomfort',

        'palpitation',
        'chest palpitation',
        'tachycardia',

        'lower chest wall pain',
        'pain, chest wall',
        'pain, chest wall, rt',
        'chest wall pain left',
        'chest wall pain right'
    }


    respiratorios = {
        'dyspnea',
        'cough',
        'hemoptysis',
        'desaturation',
        'orthopnea',
        'hyperventilation',
        'ptx - pneumothorax',
        'voice changes',
        'sorethroat',
        'throat pain',
        'discomfort, throat'
    }


    gastro = {
        'abdo pain',
        'epigastric pain',
        'abdominal discomfort',

        'ruq pain',
        'rlq pain',
        'llq pain',
        'right lower abdominal pain',
        'right lower quadrant abdominal pain',

        'upper abdominal pain',
        'lower abdominal pain',
        'low abdominal pain',
        'generalized abdominal pain',

        'abd distension',

        'vomiting',
        'nausea',
        'diarrhea',

        'hematemesis',
        'hematochezia',
        'melena',

        'anal bleeding',
        'pain, periumbilical',
        'abdominal pain, periumbilical area'
    }


    trauma = {
        'open wound',

        'facial injury',
        'head injury',
        'head trauma',
        'cerebral concussion',
        
        'jaw injury',
        'jaw pain',
        'dislocation of jaw',
        'mandibular pain',
        'tmj pain',

        'ankle injury',
        'arm injury',
        'elbow injury',
        'finger injury',
        'hand injury',
        'wrist injury',
        'knee injury',
        'toe injury',
        'foot injury',
        'leg injury',
        'right leg injury',

        'eye injury',
        'eye trauma',

        'finger laceration',
        'chin laceration',
        'lip laceration',
        'eyebrow laceration',
        'face laceration',
        'hand laceration',
        'scalp laceration',
        'wrist laceration',

        'needle stick injury',
        'chemical burn',
        'burn, arm',
        'burn, face',
        'burn, hand',
        'right forearm burn',

        'scratch wound'
    }


    hemorragicos = {
        'vaginal bleeding',
        'epistaxis',
        'hematuria',
        'oral bleeding',
        'gingival bleeding',
        'avf site bleeding',
        'bleeding, gingival',
        'bleeding, knee'
    }


    infecciosos = {
        'fever',
        'cold',
        'myalgia',
        'drug allergy'
    }


    urologicos = {
        'dysuria',
        'voiding difficulty',
        'voiding failure',
        'voiding pain',
        'urinary frequency',
        'urinary symptom',
        'hematuria',
        'oliguria',
        'retention, bladder',
        'lt flank pain',
        'rt flank pain'
    }


    psiquiatricos = {
        'anxiety',
        'depression',
        'mood depression',
        'suicidal attempt',
        'suicidal ideation',
        'suicidal thoughts (finding)',
        'hallucination',
        'delusional idea',
        'alcohol smell',
        'alcohol smelling state, drunken state',
        'drug intoxication',
        'di, drug intoxication'
    }


    dermatologicos = {
        'skin rash',
        'generalized urticaria',
        'erythematous papule',
        'itching',
        'itching sensation',
        'skin eruption',
        'painful skin lesion',
        'eczema, eyelid',
        'skin defect'
    }


    # RETORNO EXATO
    if q in neurologicos:
        return "Sintomas Neurologicos Agudos"

    elif q in cardiacos:
        return "Dor Toracica/Cardiovascular"

    elif q in respiratorios:
        return "Sintomas Respiratorios"

    elif q in gastro:
        return "Sintomas Gastrointestinais"

    elif q in trauma:
        return "Trauma"

    elif q in hemorragicos:
        return "Hemorragia"

    elif q in infecciosos:
        return "Sintomas Infecciosos"

    elif q in urologicos:
        return "Sintomas Urologicos"

    elif q in psiquiatricos:
        return "Sintomas Psiquiatricos"

    elif q in dermatologicos:
        return "Sintomas Dermatologicos"

    # FALLBACK POR PALAVRA-CHAVE
    if any(term in q for term in [
        "numb", "weakness", "paresthesia", "paralysis", "aphasia",
        "seizure", "convulsion", "syncope", "dizziness", "vertigo",
        "vision", "visual", "mental", "confusion", "consciousness",
        "facial palsy"
    ]):
        return "Sintomas Neurologicos Agudos"

    if any(term in q for term in [
        "chest", "palpitation", "tachycardia", "cardiac", "heart"
    ]):
        return "Dor Toracica/Cardiovascular"

    if any(term in q for term in [
        "dyspnea", "breath", "shortness of breath", "cough",
        "sputum", "oxygen", "desaturation", "wheezing", "throat"
    ]):
        return "Sintomas Respiratorios"

    if any(term in q for term in [
        "abd", "abdominal", "epigastric", "vomit", "vomiting",
        "diarrhea", "nausea", "melena", "hematemesis", "hematochezia",
        "stomach", "gastric", "bowel"
    ]):
        return "Sintomas Gastrointestinais"

    if any(term in q for term in [
        "rash", "itch", "itching", "skin", "urticaria", "eczema",
        "papule", "eruption"
    ]):
        return "Sintomas Dermatologicos"

    if any(term in q for term in [
        "injury", "trauma", "laceration", "wound", "burn",
        "dislocation", "fracture", "sprain", "cut", "bruise",
        "jaw injury"
    ]):
        return "Trauma"

    if any(term in q for term in [
        "bleeding", "blood", "epistaxis", "hematuria"
    ]):
        return "Hemorragia"

    if any(term in q for term in [
        "fever", "chill", "infection", "infectious", "myalgia"
    ]):
        return "Sintomas Infecciosos"

    if any(term in q for term in [
        "dysuria", "urinary", "voiding", "urine", "bladder",
        "flank pain", "retention"
    ]):
        return "Sintomas Urologicos"

    if any(term in q for term in [
        "anxiety", "depression", "suicidal", "hallucination",
        "delusion", "panic", "drug intoxication", "alcohol"
    ]):
        return "Sintomas Psiquiatricos"
    
    return "Outros Clinicos"

def preprocess_and_feature_engineer(file_path):

    df = pd.read_csv(
        file_path,
        delimiter=';',
        encoding='latin1'
    )

    all_columns_initial = [
        TARGET_COLUMN,
        'Age'
    ] + VITAL_SIGNS_AND_NUMERIC + [
        'Sex',
        'Chief_complain'
    ]

    df_model = df[all_columns_initial].copy()

    # mantém sintoma específico
    df_model['Chief_complain'] = (
        df_model['Chief_complain']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # cria grupo semântico separado
    df_model['Semantic_Group'] = (
        df_model['Chief_complain']
        .apply(agrupar_queixa)
    )

    df_model['Age'] = clean_and_convert_to_numeric(
        df_model['Age']
    )

    for col in VITAL_SIGNS_AND_NUMERIC:
        df_model[col] = clean_and_convert_to_numeric(
            df_model[col]
        )

    df_model['Age'].fillna(
        df_model['Age'].median(),
        inplace=True
    )

    for col in VITAL_SIGNS_AND_NUMERIC:
        df_model[col].fillna(
            df_model[col].median(),
            inplace=True
        )

    df_model['Age_Group'] = pd.cut(
        df_model['Age'],
        bins=[0, 2, 12, 17, 59, 120],
        labels=[
            'Bebe',
            'Crianca',
            'Adolescente',
            'Adulto',
            'Idoso'
        ]
    )

    df_model.drop(columns=['Age'], inplace=True)

    df_model.dropna(inplace=True)

    df_encoded = pd.get_dummies(
        df_model,
        columns=CATEGORICAL_FEATURES,
        drop_first=True
    )

    X = df_encoded.drop(columns=[TARGET_COLUMN])
    y = df_encoded[TARGET_COLUMN]

    return X, y, df_model.index


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

    model.fit(
        X,
        y_adjusted,
        sample_weight=sample_weights
    )

    joblib.dump(model, "modelo_xgboost.pkl")
    joblib.dump(X.columns.tolist(), "features_modelo.pkl")

    return model


TODOS_SINTOMAS = {

    # ABDOMINAL
    'abdo pain',
    'abdominal discomfort',
    'abd distension',
    'epigastric pain',
    'ruq pain',
    'rlq pain',
    'llq pain',
    'right lower abdominal pain',
    'right lower quadrant abdominal pain',
    'upper abdominal pain',
    'lower abdominal pain',
    'low abdominal pain',
    'generalized abdominal pain',
    'pain, periumbilical',
    'abdominal pain, periumbilical area',

    # GASTRO
    'vomiting',
    'nausea',
    'diarrhea',
    'hematemesis',
    'hematochezia',
    'melena',
    'anal bleeding',

    # CARDIACO
    'chest pain',
    'left chest pain',
    'rt. chest pain',
    'anterior chest pain',
    'chest discomfort',
    'chest discomfort left',
    'anterior chest discomfort',
    'palpitation',
    'chest palpitation',
    'tachycardia',
    'lower chest wall pain',
    'pain, chest wall',
    'pain, chest wall, rt',
    'chest wall pain left',
    'chest wall pain right',

    # RESPIRATORIO
    'dyspnea',
    'cough',
    'hemoptysis',
    'desaturation',
    'orthopnea',
    'hyperventilation',
    'ptx - pneumothorax',
    'voice changes',
    'sorethroat',
    'throat pain',
    'discomfort, throat',

    # NEUROLOGICO
    'headache',
    'dizziness',
    'mental change',
    'dysarthria',
    'syncope',
    'loss of consciousness',
    'amnesia',
    'rt hemiparesis',
    'lt hemiparesis',
    'right motor weakness',
    'left motor weakness',
    'right side weakness',
    'right monoparesis',
    'lower extremity paraparesis',
    'facial palsy',
    'facial palsy left',
    'left facial numbness',
    'aphasia, motor',
    'seizure',
    'blurred vision',
    'visual disturbance',
    'abnormality, visual acuity',
    'decerased visual acuity',
    'left diplopia',
    'limb paresthesia',
    'numbness',
    'rt side numbness',
    'both hand numbness',
    'hand tingling sense',

    # TRAUMA
    'open wound',
    'facial injury',
    'head injury',
    'head trauma',
    'cerebral concussion',
    'ankle injury',
    'arm injury',
    'elbow injury',
    'finger injury',
    'hand injury',
    'wrist injury',
    'knee injury',
    'toe injury',
    'foot injury',
    'leg injury',
    'right leg injury',
    'eye injury',
    'eye trauma',
    'finger laceration',
    'chin laceration',
    'lip laceration',
    'eyebrow laceration',
    'face laceration',
    'hand laceration',
    'scalp laceration',
    'wrist laceration',
    'needle stick injury',
    'chemical burn',
    'burn, arm',
    'burn, face',
    'burn, hand',
    'right forearm burn',
    'scratch wound',

    # HEMORRAGICOS
    'vaginal bleeding',
    'epistaxis',
    'hematuria',
    'oral bleeding',
    'gingival bleeding',
    'avf site bleeding',
    'bleeding, gingival',
    'bleeding, knee',

    # INFECCIOSOS
    'fever',
    'cold',
    'myalgia',
    'drug allergy',

    # UROLOGICOS
    'dysuria',
    'voiding difficulty',
    'voiding failure',
    'voiding pain',
    'urinary frequency',
    'urinary symptom',
    'oliguria',
    'retention, bladder',
    'lt flank pain',
    'rt flank pain',

    # PSIQUIATRICOS
    'anxiety',
    'depression',
    'mood depression',
    'suicidal attempt',
    'suicidal ideation',
    'suicidal thoughts (finding)',
    'hallucination',
    'delusional idea',
    'alcohol smell',
    'alcohol smelling state, drunken state',
    'drug intoxication',
    'di, drug intoxication',

    # DERMATOLOGICOS
    'skin rash',
    'generalized urticaria',
    'erythematous papule',
    'itching',
    'itching sensation',
    'skin eruption',
    'painful skin lesion',
    'eczema, eyelid',
    'skin defect',

    # OUTROS IMPORTANTES
    'general weakness',
    'g/w-general weakness',
    'weakness',
    'swelling',
    'swelling, facial',
    'swelling, neck',
    'facial painful swelling',
    'facial lesion',
    'facial swelling',
    'nasal swelling',
    'foot swelling',
    'left knee swelling',
    'left wrist swelling',
    'left thigh swelling',
    'both eyelid swelling'
}

if __name__ == "__main__":

    X, y, _ = preprocess_and_feature_engineer("data.csv")

    modelo = train_model(X, y)

    print("Modelo re-treinado com sucesso!")
