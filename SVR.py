from sklearn.preprocessing import StandardScaler
X = data_completion.loc[ : ,'strain':]
Y = data_completion.loc[ : , 'stress']
Y = pd.DataFrame(Y)
sc_X = StandardScaler().fit(X)
sc_Y = StandardScaler().fit(Y)
X = sc_X.transform(X)
Y = sc_Y.transform(Y)
print('The feature values after scaling:')
print(X)
print('The target values after scaling:')
print(Y)

from sklearn.model_selection import train_test_split
# Shuffle and split the data into training and testing subsets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
print('Train_features:')
print(X_train)
Y_train = np.array(Y_train).ravel()
print('Train_target:')
print(Y_train)

from sklearn.svm import SVR
reg_rbf = SVR(C=1050, kernel='rbf', gamma=4e-3, epsilon = 0.07)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
reg_rbf.fit(X_train, Y_train)
Y_pred = reg_rbf.predict(X_test)
Y_pred = sc_Y.inverse_transform(Y_pred.reshape(-1, 1))
# output R2 score
print("R^2: {}".format(reg_rbf.score(X_test, Y_test)))
# output root mean squared error
rmse = np.sqrt(mean_squared_error(sc_Y.inverse_transform(Y_test), Y_pred))
print("Root Mean Squared Error: {}".format(rmse))
mae = mean_absolute_error(sc_Y.inverse_transform(Y_test), Y_pred)
print("Mean Absolute Error: {}".format(mae))

# Try different values for C from 800 to 1200, the interval is 50
c_range = np.arange(800,1200,50)
c_scores = []
for c in c_range:
    reg = SVR(C=c, kernel='rbf', gamma=4e-3, epsilon = 0.07)
    score = cross_val_score(reg,X_train,Y_train, cv=5, scoring='neg_mean_squared_error')
    score = sc_Y.inverse_transform(-score)
    score = np.sqrt(score)
    c_scores.append(score.mean())
print(c_scores)

