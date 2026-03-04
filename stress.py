import pandas as pd
import seaborn as sns 
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv",sep=',', header=0)

df.info()
df = df.drop(columns=['User_ID'])
df['Device_Type'] = df['Device_Type'].map({'iOS':1,'Android':0})
df['Gender'] = df['Gender'].map({'Female':2,'Male':1,'Other':0})

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

X=df_final.drop('Stress_Level',axis=1)
Y=df_final['Stress_Level']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
modele_lr =LinearRegression()
modele_lr.fit(X_train,Y_train)
Y_pred= modele_lr.predict(X_test)# Prend la prédiction
Mse=mean_squared_error(Y_test,Y_pred)
r2 = r2_score(Y_test,Y_pred)

print(f'mean squared error equals: {Mse:.4f}')
print(f'r2 : {r2:.4f}')


