import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os

# 1. تحميل البيانات
df = pd.read_csv('Mall_Customers.csv')

# إعادة تسمية الأعمدة لتسهيل التعامل معها
df.columns = ['CustomerID', 'Gender', 'Age', 'Annual_Income', 'Spending_Score']

# 2. تجهيز البيانات (Preprocessing & Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])

# 3. تدريب نموذج التجميع (Clustering - K-Means)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. تدريب نموذج الانحدار (Regression - Predict Spending Score)
X_reg = df[['Age', 'Annual_Income']]
y_reg = df['Spending_Score']
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_reg, y_reg)

# 5. تدريب نموذج التصنيف (Classification - High vs Low Spender)
# نعتبر العميل High Spender (1) إذا كان Spending Score أكبر من 50
df['Target_Class'] = (df['Spending_Score'] > 50).astype(int)
X_clf = df[['Age', 'Annual_Income']]
y_clf = df['Target_Class']
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_clf, y_clf)

# 6. إنشاء مجلد models وحفظ النماذج والـ Scaler
os.makedirs('models', exist_ok=True)

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(kmeans, 'models/cluster_model.pkl')
joblib.dump(reg_model, 'models/regression_model.pkl')
joblib.dump(clf_model, 'models/classification_model.pkl')

print("All models trained and saved successfully in 'models/' directory!")
