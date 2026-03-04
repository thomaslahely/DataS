import pandas as pd
import seaborn as sns 
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv",sep=',', header=0)

df.info()
df = df.drop(columns=['User_ID'])

nominal_cols = df.select_dtypes(include='object').columns

df_final = pd.get_dummies(df, columns=nominal_cols, drop_first=True)


# Calcul des corrélations
correlation_matrix= df_final.select_dtypes(exclude=['object']).corr()

# Supprimer Stress_Level et les NaN
correlation_objectif = correlation_matrix['Stress_Level'].drop('Stress_Level').sort_values(ascending=False)

# Trier

plt.figure(figsize=(8,6))
sns.barplot(x=correlation_objectif.values, y=correlation_objectif.index)
plt.title("Corrélation avec le stress")
plt.show()
