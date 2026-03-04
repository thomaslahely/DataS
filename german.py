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
import joblib



scaler = StandardScaler()
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_commitment",
    "personal_status", "other_parties", "residence_since", "property_magnitude",
    "age", "other_payment_plans", "housing", "existing_credits",
    "job", "num_dependents", "own_telephone", "foreign_worker", "target"
]

df = pd.read_csv("german.data.csv", sep=" ", header=None, names=COLUMNS)

# On regarde les valeurs pour savoir pourquoi les gens n'arrivent pas à rembourser leurs crédits 
# crédits à risque à crédits sains
df.info()
print(df["target"].value_counts())# permet de voir les crédits sains et à risques

# On passe en binaire ces données pour les manipuler plus facilement car elles n'ont que deux valeurs possibles
df['target'] = df['target'].map({1: 0, 2: 1})
df['foreign_worker'] = df['foreign_worker'].map({'A201':1,'A202':0})
df['own_telephone'] = df['own_telephone'].map({'A191':1,'A192':0})

# On récupere les données qui ne sont pas int 
nominal_cols = df.select_dtypes(include='object').columns

# One-Hot Encoding
df_final = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

correlation_matrix= df_final.select_dtypes(exclude=['object']).corr()

correlation_objectif = correlation_matrix['target'].drop('target').sort_values(ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=correlation_objectif.values, y=correlation_objectif.index)
plt.title("Corrélation avec le risque")
plt.show()

df_final['monthly_installment_estimation'] = df_final['credit_amount'] / df_final['duration']

X=df_final.drop('target',axis=1)# les indices à notre disposition et axis pour drop la colonne de target
Y=df_final['target'] # la réponse


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
modele_risque =LogisticRegression()
modele_risque.fit(X_train_scaled,Y_train)
y_pred = modele_risque.predict(X_test_scaled)


cm = confusion_matrix(Y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sain', 'Risque'])
disp.plot(cmap='Blues')
plt.title("Matrice de Confusion - Modèle de Risque")
plt.show()

#randomForest
"""
n_estimators : il s’agit du nombre d’arbres dans la forêt
criterion : il s’agit du critère utilisé pour construire les arbres et séparer les branches des arbres
max_depth : il s’agit de la profondeur maximale des arbres utilisés (le nombre de niveaux dans l’arbre de décision)
min_sample_split : il s’agit du nombre d’échantillons minimal dans une feuille pour refaire une séparation
min_samples_leaf : il s’agit du nombre d’échantillons minimal pour créer une feuille
min_weight_fraction_leaf : il s’agit de la fraction du nombre total d’échantillon minimal pour créer une feuille
max_features : il s’agit du nombre de colonnes sélectionnées pour chaque arbre (par défaut on prend la racine carré du nombre de colonnes)
max_leaf_nodes : il s’agit du nombre maximal de feuilles
min_impurity_decrease : il s’agit de la baisse minimale du critère d’impureté pour faire une séparation
bootstrap : paramètre pour utiliser du bootstrap, si il est à False, le même échantillon est pris pour chaque arbre
n_jobs ; nombre de traitements à effectuer en parallèle
random_state : graine aléatoire
warm_start : ceci permet de repartir du résultat du dernier apprentissage pour faire l’apprentissage
class_weights : il s’agit des poids associés à chaque classe si cela a un sens
max_samples : si vous voulez réduire le nombre d’observations dans vos échantillons bootstrap
"""
modele_rf = RandomForestClassifier(
     n_estimators=100,
     criterion='gini',
     max_depth=None,
     min_samples_split=2,
     min_samples_leaf=1,
     min_weight_fraction_leaf=0.0,
     max_features="sqrt",
     max_leaf_nodes=None,
     min_impurity_decrease=0.0,
     bootstrap=True,
     oob_score=False,
     n_jobs=None,
     random_state=42,
     verbose=0,
     warm_start=False,
     class_weight=None,
     ccp_alpha=0.0,
     max_samples=None,)

modele_rf.fit(X_train, Y_train)
y_pred_rf = modele_rf.predict(X_test)

print("--- Rapport Random Forest ---")
print(classification_report(Y_test, y_pred_rf))

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, Y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("--- Rapport XGBoost ---")
print(classification_report(Y_test, y_pred_xgb))

modeles_statistiques= []


def add_model_stats(model_name, y_true, y_pred):
    stats = {
        'Modèle': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    modeles_statistiques.append(stats)

add_model_stats('Logistic Regression', Y_test, y_pred)      # La régression logistique
add_model_stats('Random Forest', Y_test, y_pred_rf)      # Random Forest
add_model_stats('XGBoost', Y_test, y_pred_xgb)            # XGBoost


df_comparison = pd.DataFrame(modeles_statistiques).sort_values(by='Recall', ascending=False)

print("--- Comparaison des Modèles---")
print(df_comparison)


explainer = shap.LinearExplainer(modele_risque, X_train_scaled, feature_names=X.columns)
shap_values = explainer.shap_values(X_test_scaled)

plt.title("SHAP")
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

joblib.dump(modele_rf,'modele_randomforest.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')