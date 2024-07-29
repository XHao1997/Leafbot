#In[1]

import numpy as np
import sklearn
#In[2]
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import LinearRegression
import joblib
y = np.load('joint1_nn_y_1.npy')
X = np.load('joint1_nn_x_1.npy')
y=y[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)

estimators = [('ridge', RidgeCV()),
              ('lasso', LassoCV(random_state=42)),
              ('knr',make_pipeline(StandardScaler(), SVR(C=20, epsilon=0.1)))]
#In[3]
from sklearn.ensemble import StackingRegressor
# final_estimator = GradientBoostingRegressor(
#     n_estimators=100, subsample=0.5, min_samples_leaf=25, max_features=1,
#     random_state=42)
# final_estimator = make_pipeline(StandardScaler(), SVR(C=20, epsilon=1e-2))
final_estimator = LinearRegression()

reg_j1 = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator)
reg_j1.fit(X_train,y_train)
joblib.dump(reg_j1, 'weights/ereg_j1.pkl')
model = joblib.load('weights/ereg_j1.pkl')