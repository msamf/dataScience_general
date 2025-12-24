import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

## DATA LOADING
path_to_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

col_names = [
    'age', 'workclass', 'fnlwgt','education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain','capital-loss',
    'hours-per-week','native-country', 'income'
]

df = pd.read_csv(path_to_data, header=None, names = col_names)
print(df.head())

# clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()

target_column = "income"
raw_feature_cols = [
    'age',
    'education-num',
    'workclass',
    'hours-per-week',
    'sex',
    'race'
]

# look at target column
print(df['income'].value_counts(normalize=True))

# look at feature columns
print(df[raw_feature_cols].dtypes)


## DATA PREPARATION
# convert all data to numerical
X = pd.get_dummies(df[raw_feature_cols], drop_first=True)
print(X.head())

# convert target variable to binary
y = df["income"].apply(lambda x: 1 if x == "<=50K" else 0)

# train-est split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


## MODEL INITIATION AND TRAINING
# create AdaBoost classifier
decision_stump = DecisionTreeClassifier(max_depth=1)
ada_classifier = AdaBoostClassifier(estimator=decision_stump)

# create GradientBoost Classifier
grad_classifier = GradientBoostingClassifier()

# fit models and get predictions
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)

grad_classifier.fit(X_train, y_train)
y_pred_grad = grad_classifier.predict(X_test)

## MODEL EVALUATION 
ada_accuracy = accuracy_score(y_test, y_pred_ada)
ada_f1 = f1_score(y_test, y_pred_ada)
print(f"Ada: accuracy of {ada_accuracy}, and f1 score of {ada_f1}")

grad_accuracy = accuracy_score(y_test, y_pred_grad)
grad_f1 = f1_score(y_test, y_pred_grad)
print(f"GradBoost: accuracy of {grad_accuracy}, and f1 score of {grad_f1}")

## HYPERPARAMETER TUNING
n_estimators_list = [10, 30, 50, 70, 90]

est_params = {"n_estimators": n_estimators_list}
ada_gridsearch = GridSearchCV(ada_classifier, est_params, cv=5, scoring="accuracy", verbose=True)
ada_gridsearch.fit(X_train, y_train)

# plot mean test scores
ada_scores_list = ada_gridsearch.cv_results_["mean_test_score"]
plt.scatter(n_estimators_list, ada_scores_list)
plt.xlabel("n_estimators")
plt.ylabel("mean_test_score")
plt.savefig("estimators_vs_score.png")

# output best model
print("Based on the graph, the best accuracy is achieved with n_estimators = 90. However, the accuracy is very similar.")
print("Overall, n_estimators = 50 or 70 should be chosen, as it balances a high accuracy score with computational efficiency.")
