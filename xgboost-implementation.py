import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

df = pd.read_csv('matches_data.csv')  

X = df.drop(columns=['Date', 'Team1', 'Team2', 'Score'])

y = df['Score']
print(sum(y) / len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


print(sum(y_train) / len(y_train))
print(sum(y_test) / len(y_test))
#round 1
#param_grid = {
#    'max_depth': [3, 4, 5],
#    'learning_rate': [0.1, 0.01, 0.05],
#    'gamma': [0, 0.25, 1.0],
#    'reg_lambda': [0.1, 1.0, 10.0],
#    'scale_pos_weight': [1, 3, 5]
#}


#round 2
param_grid = {
    'max_depth': [4],
    'learning_rate': [0.05],
    'gamma': [1.0],
    'reg_lambda': [1.0],
    'scale_pos_weight': [5]
}
#colsample_bytree parameter for overfitting
#subsample parameter for overfitting
#optimal_params = GridSearchCV(
#    estimator=XGBClassifier(objective='binary:logistic', seed=42, eval_metric='aucpr',use_label_encoder=False, early_stopping_rounds=10),
#    param_grid=param_grid,
#    scoring='roc_auc',
#    verbose=0,
#    cv=3
#)


#optimal_params.fit(X_train,
#            y_train,
#            eval_set=[(X_test, y_test)])

#print(optimal_params)


clf_xgb = XGBClassifier(objective='binary:logistic', seed=42, eval_metric='aucpr',use_label_encoder=False, early_stopping_rounds=10)


ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test, display_labels=['Loss', 'Win'])
plt.show()
y_pred = clf_xgb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
