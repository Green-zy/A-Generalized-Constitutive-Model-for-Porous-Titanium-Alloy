from sklearn.model_selection import train_test_split
# create feature data and target data without scaling
X = data_completion.loc[ : ,'strain':]
Y = data_completion.loc[ : , 'stress']
X = np.array(X)
Y = np.array(Y).ravel()
# Shuffle and split the data into training and testing subsets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
print('Train_features:')
print(X_train)
Y_train = np.array(Y_train).ravel()
print('Train_target:')
print(Y_train)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
reg_tree = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=50, learning_rate=1)
reg_tree.fit(X_train, Y_train)
Y_pred = reg_tree.predict(X_test)
# calculate the training and testing scores
print("R^2: {}".format(reg_tree.score(X_test, Y_test)))
mae = np.sqrt(mean_absolute_error(Y_test, Y_pred))
print("Mean Absolute Error: {}".format(mae))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
# create 10 groups of cross-validation sets 
cro_val = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# set figure size
train_fig = plt.figure(figsize=(14,12))
# train&test models with different max_depth, n_estimators and learning_rate take default values
perf_scores = []
perf_maes = []
depth = [3,5,10,30,50,70,80]
for x in depth:
    reg_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=x), n_estimators=50, learning_rate=1)
    # calculate the training and testing scores
    perf_score = cross_val_score(reg_2,X_train,Y_train, cv=10, scoring='r2')              
    perf_mae = cross_val_score(reg_2,X_train,Y_train, cv=10, scoring='neg_mean_absolute_error')
    perf_scores.append(perf_score.mean())
    perf_maes.append(-perf_mae.mean())   
# subplot the learning curves 
perf_r1 = train_fig.add_subplot(2, 1, 1)
perf_r1.plot(depth, perf_scores, marker='o', color = 'blue')
plt.xlabel('max_depth')
plt.ylabel('r2 score')
for a, b in zip(depth, perf_scores):
    plt.text(a, b, '%.6f'%b, ha='center', va='bottom', fontsize=10)
plt.title('R^2 score')
perf_r2 = train_fig.add_subplot(2, 1, 2)
perf_r2.plot(depth, perf_maes, marker='o', color = 'red')
plt.xlabel('max_depth')
plt.ylabel('MAE')
for a, b in zip(depth, perf_maes):
    plt.text(a, b, '%.6f'%b, ha='center', va='bottom', fontsize=10)
plt.title('Mean Absolute Error')
plt.subplots_adjust(hspace=0.3)
plt.show()

from sklearn.model_selection import GridSearchCV
import warnings
warnings.simplefilter('ignore')
depth_space = np.linspace(45, 55, 11)
opt = GridSearchCV(DecisionTreeRegressor(), {'max_depth': depth_space})
opt.fit(X_train, Y_train)
print("Tuned DTR Parameters: {}".format(opt.best_params_)) 

n_estimators_space = np.linspace(20, 60, 41).astype(int)
learning_rate_space = np.linspace(0.1, 1, 19)
opt_2 = GridSearchCV(AdaBoostRegressor(DecisionTreeRegressor(max_depth=53)), param_grid={"n_estimators": n_estimators_space, 
                                                                                        "learning_rate": learning_rate_space})
opt_2.fit(X_train, Y_train)
print("Tuned ABR Parameters: {}".format(opt_2.best_params_))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 40)
reg_opt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=53), n_estimators=54, learning_rate=0.2)
reg_opt.fit(X_train, Y_train)
Y_pred = reg_tree.predict(X_test)
# calculate the training and testing scores
print("R^2: {}".format(reg_tree.score(X_test, Y_test)))
mae = np.sqrt(mean_absolute_error(Y_test, Y_pred))
print("Mean Absolute Error: {}".format(mae))

# construct the final model
reg_final = AdaBoostRegressor(DecisionTreeRegressor(max_depth=53), n_estimators=54, learning_rate=0.2)
reg_final.fit(X, Y)
final_test_targer = data_completion.iloc[28168:,0]
final_test_targer = np.array(final_test_targer).ravel()
final_test_features = data_completion.iloc[28168:,1:]
final_test_features = np.array(final_test_features)
final_pred = reg_final.predict(final_test_features)
# calculate accuracy
ite = list(final_test_targer)
ind_n = 0
err = 0
err_perc = 0
for val in ite:
    err += math.fabs(val - final_pred[ind_n])
    if val > 0:
        err_perc += math.fabs(val - final_pred[ind_n]) / val
    if val < 0:
        err_perc += math.fabs(val - final_pred[ind_n]) / (-val)
    if val == 0:
        continue   
    ind_n += 1
err /=  ind_n
err_perc = 100*err_perc/ind_n
print("The Mean Absolute Error: {}".format(err))
print("The Mean Absolute Error Percentage: {}%".format("%.2f" % err_perc))

final_test_strain = data_completion.iloc[28168:,1]
final_test_strain = np.array(final_test_strain)
final_fig = plt.figure(figsize=(12,9))
final_plot = final_fig.add_subplot(1, 1, 1)
plt.plot(final_test_strain, final_test_targer, lw=3, color='orange', label='true stress')
plt.plot(final_test_strain, final_pred, lw=3, color='blue',  linestyle=':', label='predicted stress')
plt.title('Stress-strain curves (T=25℃, strain rate=2300/s, porosity=26%)')
plt.xlabel('strain')
plt.ylabel('stress')
plt.legend(loc='lower right') 
plt.show()

from mpl_toolkits.mplot3d import axes3d
# create feature values
reg_finalv1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=53, min_samples_leaf=4), n_estimators=54, learning_rate=0.2)
reg_finalv1.fit(X, Y)
strain_g1 = []
strain_g1 = np.arange(0, 0.354, 0.005)
strainrate_g1 = []
porosity_g1 = []
T_g1_1 = []
T_g1_2 = []
T_g1_3 = []
T_g1_4 = []
T_g1_5 = []
# create feature values
for temp_g1 in strain_g1:
    strainrate_g1.append(2380)  
    porosity_g1.append(27.5)
    T_g1_1.append(20)
    T_g1_2.append(80)
    T_g1_3.append(160)
    T_g1_4.append(220) 
    T_g1_5.append(280) 
g1_1 = pd.DataFrame({'test_strain':strain_g1, 'test_T':T_g1_1, 'test_strainrate':strainrate_g1, 'test_porosity':porosity_g1})
g1_2 = pd.DataFrame({'test_strain':strain_g1, 'test_T':T_g1_2, 'test_strainrate':strainrate_g1, 'test_porosity':porosity_g1})
g1_3 = pd.DataFrame({'test_strain':strain_g1, 'test_T':T_g1_3, 'test_strainrate':strainrate_g1, 'test_porosity':porosity_g1})
g1_4 = pd.DataFrame({'test_strain':strain_g1, 'test_T':T_g1_4, 'test_strainrate':strainrate_g1, 'test_porosity':porosity_g1})
g1_5 = pd.DataFrame({'test_strain':strain_g1, 'test_T':T_g1_5, 'test_strainrate':strainrate_g1, 'test_porosity':porosity_g1})
pred_g1_1 = reg_finalv1.predict(g1_1)
pred_g1_2 = reg_finalv1.predict(g1_2)
pred_g1_3 = reg_finalv1.predict(g1_3)
pred_g1_4 = reg_finalv1.predict(g1_4)
pred_g1_5 = reg_finalv1.predict(g1_5)
# set plot for prediction
fig_pred1 = plt.figure(figsize=(13,9))
ax_pre1 = fig_pred1.gca(projection='3d')
ax_pre1.plot(strain_g1, pred_g1_1, zs=20, zdir='y', color = 'b')
ax_pre1.plot(strain_g1, pred_g1_2, zs=80, zdir='y', color = 'b')
ax_pre1.plot(strain_g1, pred_g1_3, zs=160, zdir='y', color = 'b')
ax_pre1.plot(strain_g1, pred_g1_4, zs=220, zdir='y', color = 'b')
ax_pre1.plot(strain_g1, pred_g1_5, zs=280, zdir='y', color = 'b')
ax_pre1.set_xlabel('Strain')
ax_pre1.set_ylabel('Temperature(℃)')
ax_pre1.set_zlabel('Stress(MPa)')
ax_pre1.set_title("Stress-strain curves \n (strain rate = 2380, porosity = 27.5%) ")
ax_pre1.view_init(elev=35., azim=-60)
plt.show()

# create feature values
reg_finalv = AdaBoostRegressor(DecisionTreeRegressor(max_depth=53, min_samples_leaf=2), n_estimators=54, learning_rate=0.2)
reg_finalv.fit(X, Y)
strain_g3 =[]
strain_g3 = np.arange(0, 0.06, 0.005)
T_g3 = []
strainrate_g3 = []
porosity_g3_1 = []
porosity_g3_2 = []
porosity_g3_3 = []
# create feature values
for temp_g1 in strain_g3:
    T_g3.append(120)  
    strainrate_g3.append(550)
    porosity_g3_1.append(5)
    porosity_g3_2.append(25)
    porosity_g3_3.append(45)
g3_1 = pd.DataFrame({'test_strain':strain_g3, 'test_T':T_g3, 'test_strainrate':strainrate_g3, 'test_porosity':porosity_g3_1})
g3_2 = pd.DataFrame({'test_strain':strain_g3, 'test_T':T_g3, 'test_strainrate':strainrate_g3, 'test_porosity':porosity_g3_2})
g3_3 = pd.DataFrame({'test_strain':strain_g3, 'test_T':T_g3, 'test_strainrate':strainrate_g3, 'test_porosity':porosity_g3_3})
pred_g3_1 = reg_finalv.predict(g3_1)
pred_g3_2 = reg_finalv.predict(g3_2)
pred_g3_3 = reg_finalv.predict(g3_3)
# set plot for prediction
fig_pred1 = plt.figure(figsize=(13,9))
ax_pre1 = fig_pred1.gca(projection='3d')
ax_pre1.plot(strain_g3, pred_g3_1, zs=5, zdir='y', color = 'b')
ax_pre1.plot(strain_g3, pred_g3_2, zs=25, zdir='y', color = 'b')
ax_pre1.plot(strain_g3, pred_g3_3, zs=45, zdir='y', color = 'b')
ax_pre1.set_xlabel('Strain')
ax_pre1.set_ylabel('Porosity(%)')
ax_pre1.set_zlabel('Stress(MPa)')
ax_pre1.set_title("Stress-strain curves \n (T = 120℃, strain rate = 550) ")
ax_pre1.view_init(elev=35., azim=-60)
plt.show()

# create feature values
reg_finalv2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=53, min_samples_leaf=2), n_estimators=54, learning_rate=0.2)
reg_finalv2.fit(X, Y)
strain_g2_1 = np.arange(0, 0.135, 0.005)
strain_g2_3 = np.arange(0, 0.235, 0.005)
strain_g2_4 = np.arange(0, 0.345, 0.005)
T_g2_1 = []
T_g2_3 = []
T_g2_4 = []
porosity_g2_1 = []
porosity_g2_3 = []
porosity_g2_4 = []
strainrate_g2_1 = []
strainrate_g2_2 = []
strainrate_g2_3 = []
strainrate_g2_4 = []
strainrate_g2_5 = []
# create feature values
for temp_g1 in strain_g2_1:
    T_g2_1.append(315)  
    porosity_g2_1.append(25.5)
    strainrate_g2_1.append(1000)
    strainrate_g2_2.append(1500)
for temp_g1 in strain_g2_3:
    T_g2_3.append(315)  
    porosity_g2_3.append(25.5)
    strainrate_g2_3.append(2000)
for temp_g1 in strain_g2_4:
    T_g2_4.append(315)  
    porosity_g2_4.append(25.5)
    strainrate_g2_4.append(2500) 
    strainrate_g2_5.append(3000) 
g2_1 = pd.DataFrame({'test_strain':strain_g2_1, 'test_T':T_g2_1, 'test_strainrate':strainrate_g2_1, 'test_porosity':porosity_g2_1})
g2_2 = pd.DataFrame({'test_strain':strain_g2_1, 'test_T':T_g2_1, 'test_strainrate':strainrate_g2_2, 'test_porosity':porosity_g2_1})
g2_3 = pd.DataFrame({'test_strain':strain_g2_3, 'test_T':T_g2_3, 'test_strainrate':strainrate_g2_3, 'test_porosity':porosity_g2_3})
g2_4 = pd.DataFrame({'test_strain':strain_g2_4, 'test_T':T_g2_4, 'test_strainrate':strainrate_g2_4, 'test_porosity':porosity_g2_4})
g2_5 = pd.DataFrame({'test_strain':strain_g2_4, 'test_T':T_g2_4, 'test_strainrate':strainrate_g2_5, 'test_porosity':porosity_g2_4})
pred_g2_1 = reg_finalv2.predict(g2_1)
pred_g2_2 = reg_finalv2.predict(g2_2)
pred_g2_3 = reg_finalv2.predict(g2_3)
pred_g2_4 = reg_finalv2.predict(g2_4)
pred_g2_5 = reg_finalv2.predict(g2_5)
# set plot for prediction
fig_pred1 = plt.figure(figsize=(13,9))
ax_pre1 = fig_pred1.gca(projection='3d')
ax_pre1.plot(strain_g2_1, pred_g2_1, zs=1000, zdir='y', color = 'b')
ax_pre1.plot(strain_g2_1, pred_g2_2, zs=1500, zdir='y', color = 'b')
ax_pre1.plot(strain_g2_3, pred_g2_3, zs=2000, zdir='y', color = 'b')
ax_pre1.plot(strain_g2_4, pred_g2_4, zs=2500, zdir='y', color = 'b')
ax_pre1.plot(strain_g2_4, pred_g2_5, zs=3000, zdir='y', color = 'b')
ax_pre1.set_xlabel('Strain')
ax_pre1.set_ylabel('Strain rate(/s)')
ax_pre1.set_zlabel('Stress(MPa)')
ax_pre1.set_title("Stress-strain curves \n (T = 195℃, porosity = 25.5%) ")
ax_pre1.view_init(elev=35., azim=-60)
plt.show()
