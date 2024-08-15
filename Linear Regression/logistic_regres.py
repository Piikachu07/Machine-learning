import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score

x = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(x,y)

# log_odds = logr.coef_
# odds = np.exp(log_odds)
# print(odds)

y_pred = logr.predict(x)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy of the model is {accuracy: .2f} ")



# import numpy as np
# from sklearn import linear_model

# x= np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 888]).reshape(-1,1)
# y= np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# logr = linear_model.LogisticRegression()
# logr.fit(x,y)

# def logit2prob(logr, x):
#     log_odds = logr.coef_ *x + logr.intercept_
#     odds = np.exp(log_odds)
#     probab = odds/(odds+1)
#     return probab

# print(logit2prob(logr,x))

