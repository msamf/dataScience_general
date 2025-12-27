import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

## Import data
dataset = pd.read_csv("life_expectancy.csv")

# look at data
print("FIRST FIVE ENTRIES \n", dataset.head(), "\n\n")
print("INFO \n", dataset.info(), "\n\n")
print("DESCRIBE \n", dataset.describe(), "\n\n")

# drop Country column as it shouldn't be used in the model
dataset = dataset.drop(["Country"], axis=1) 

## Create labels and features
labels = dataset.iloc[:, -1] # select last column as labels
features = dataset.iloc[:, :-1] # select last column as labels

features = pd.get_dummies(features) # encode categorical data

## Split into test and train
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=14)

## Standardize numerical features
numerical_features = features.select_dtypes(include=["float64", "int64"])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder="passthrough")
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

## Build model
my_model = Sequential()
input = InputLayer(input_shape=(features.shape[1], ))
my_model.add(input) # input layer
my_model.add(Dense(64, activation="relu")) # hidden layer of 64 units
my_model.add(Dense(1)) # output layer
print(my_model.summary())

## Compile model
op = Adam(learning_rate=0.01) # optimizer
my_model.compile(loss="mse", metrics=["mae"], optimizer=op)

## Fit and evaluate model
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)
print("MSE: ", res_mse, " // MAE: ", res_mae)