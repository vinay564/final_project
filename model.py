import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


def encodeDatasets(df, columns):
    for i, column in enumerate(columns):
        lb = LabelEncoder()
        df[column] = lb.fit_transform(df[column])
        print(column,lb.classes_)
    return df


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Building the model : neural network with L1 regularization
def MyModel(name="RFR"):
    if name == 'RFR':
        model = RandomForestRegressor(n_estimators=100)
    elif name == "ridge":
        model = Ridge()
    else:
        model = KNeighborsRegressor()
    return model


def evaluation(model, X_test, y_test):
    # MSE and r squared values
    y_pred = model.predict(X_test)
    print("r2 score:", round(r2_score(y_test, y_pred), 4))
    print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
    # Scatter plot of predicted values
    plt.scatter(y_test, y_pred, s=2, alpha=0.7)
    plt.plot(list(range(2, 8)), list(range(2, 8)), color='black', linestyle='--')
    plt.title('Predicted vs. actual values of Test set')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.show()
