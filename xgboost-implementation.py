import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

df = pd.read_csv('matches_data.csv')  

X = df.drop(columns=['Date', 'Team1', 'Team2', 'Score'])

y = df['Score']
print(sum(y) / len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


print(sum(y_train) / len(y_train))
print(sum(y_test) / len(y_test))


clf_xgb = XGBClassifier(objective='binary:logistic', seed=42, eval_metric='aucpr',use_label_encoder=False, early_stopping_rounds=10)

clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            eval_set=[(X_test, y_test)])

ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test, display_labels=['Loss', 'Win'])
plt.show()
y_pred = clf_xgb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
