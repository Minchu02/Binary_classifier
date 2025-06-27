import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


df = pd.read_csv("data.csv")
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

df = df.drop(columns=['id'], errors='ignore')
df = df.drop(columns=['Unnamed: 32'], errors='ignore')  

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']


print("\n NaN check:\n", X.isnull().sum())
print("\n Infinite check:\n", np.isinf(X).sum())

X = X.fillna(0)  # or: X = X.dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("ROC AUC:  ", roc_auc_score(y_test, y_prob))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.3f})".format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

threshold = 0.6
y_pred_tuned = (y_prob >= threshold).astype(int)

cm_tuned = confusion_matrix(y_test, y_pred_tuned)
print(f"\nWith Threshold = {threshold}")
print("Confusion Matrix:\n", cm_tuned)
print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall:   ", recall_score(y_test, y_pred_tuned))
print("F1 Score: ", f1_score(y_test, y_pred_tuned))

