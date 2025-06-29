
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt




df = pd.read_csv('personal_productivity_dataset.csv')
print(df.dtypes)
print(df.head())
X = df.drop('Productive', axis=1)
y = df['Productive']


X = pd.get_dummies(X, columns=['Weather'], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Feature Importance")
plt.show()


def recommend_tip(sample):
    tip = ""
    if sample['SleepHours'] < 6.5:
        tip += "Try to sleep at least 7 hours. "
    if sample['ScreenTime'] > 6:
        tip += "Reduce screen time to stay focused. "
    if sample['ExerciseMins'] < 15:
        tip += "Add some exercise to improve energy. "
    if sample['Meals'] < 3:
        tip += "Maintain regular meals. "
    if tip == "":
        tip = "You're on track! Keep the routine."
    return tip


sample = X_test.iloc[0:1]
prediction = model.predict(sample)[0]
predicted_label = "Productive" if prediction == 1 else "Not Productive"


print(f"\n🧠 Predicted: \n{predicted_label}")
print("💡 Tip:", recommend_tip(sample.iloc[0]))
