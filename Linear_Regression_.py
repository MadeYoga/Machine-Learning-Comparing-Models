
import pandas as pd
import os

abspath             = os.path.abspath(__file__)
this_script_path    = os.path.dirname(abspath)
datasets_path       = this_script_path + "\\Datasets"

os.chdir(datasets_path)

training_data = pd.read_csv("heartdisease-train.csv")
X_train = training_data.loc[:, training_data.columns[1:]]
y_train = training_data.loc[:, training_data.columns[-1:]]

test_data = pd.read_csv("heartdisease-test.csv")
X_test = test_data.loc[:, test_data.columns[:-1]]
y_test = test_data.loc[:, test_data.columns[-1:]]

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)

predicted_y = linreg.predict(X_test)

print(predicted_y, y_test)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
print(mean_squared_error(predicted_y, y_test))
scores = cross_val_score(
            linreg, 
            X=X_train, 
            y=y_train.values.ravel(), 
            cv=10, 
            scoring='neg_mean_squared_error'
            )
print(scores.mean() * 100)