import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
try:
    print("Loading dataset...")
    df = pd.read_csv('C:/Users/faree/Downloads/projects/fraud detection/creditcard.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("⚠️ Error: Dataset file not found. Please check the file path.")
    exit()

print("Initial shape:", df.shape)
print("Class distribution:\n", df['Class'].value_counts())
print("Scaling 'Amount' and 'Time' columns...")
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)
y = df['Class']
print("Applying SMOTE (this may take a minute)...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("Resampled class distribution:\n", y_res.value_counts())
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Making predictions...")
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show(block=True)

