from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('modele_randomforest.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('model_columns.pkl')

class CreditInput(BaseModel):
    #valeur par défaut pour les rendre optionnel 
    checking_status: str = "A11" 
    duration: int = 0
    credit_history: str = "A32"
    purpose: str = "A40"
    credit_amount: int = 0
    savings_status: str = "A61"
    employment: str = "A71"
    installment_commitment: int = 0
    personal_status: str = "A91"
    other_parties: str = "A101"
    residence_since: int = 0
    property_magnitude: str = "A121"
    age: int = 0
    other_payment_plans: str = "A143"
    housing: str = "A151"
    existing_credits: int = 0
    job: str = "A171"
    num_dependents: int = 0
    own_telephone: str = "A191"
    foreign_worker: str = "A201"

@app.post("/prediction")

def prediction(data: CreditInput):
    df= pd.DataFrame([data.dict()])

    df['foreign_worker'] = df['foreign_worker'].map({'A201':1,'A202':0})
    df['own_telephone'] = df['own_telephone'].map({'A191':1,'A192':0})
    df['monthly_installment_estimation'] = df['credit_amount'] / df['duration']

    nominal_cols = df.select_dtypes(include='object').columns
    df_1 = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # Si l'utilisateur ne remplit pas toutes les colonnes on met zéro
    # et comme le modèle est entraîné sur un nb attendu 
    df_final= df_1.reindex(columns=columns, fill_value=0)
    prediction = model.predict(df_final)
    probability = model.predict_proba(df_final)[:, 1] # on récupere juste la chance qu'il soit à risque

    return {
        "prediction": "Risque" if int(prediction[0]) == 1 else "Sain",
        "probability": round(float(probability[0]), 2)
    }



