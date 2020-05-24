# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 01:24:34 2020

@author: Yun Zhou
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
X = data_completion.loc[ : ,'strain':]
Y = data_completion.loc[ : , 'stress']
X = np.array(X)
Y = np.array(Y).ravel()

reg_denoise = AdaBoostRegressor(DecisionTreeRegressor(max_depth=53, min_samples_leaf=4), n_estimators=54, learning_rate=0.2)
reg_denoise.fit(X, Y)

strain_0_20_500 = []
strain_0_20_500 = np.arange(0, 0.38, 0.0025)
strainrate_0_20_500 = []
porosity_0_20_500 = []
T_0_20_500 = []
for temp_0_20_500 in strain_0_20_500:
    strainrate_0_20_500.append(500)
    porosity_0_20_500.append(0)
    T_0_20_500.append(20)
features_0_20_500 = pd.DataFrame({'strain':strain_0_20_500, 'T':T_0_20_500, 'strainrate':strainrate_0_20_500, 'porosity':porosity_0_20_500})
pred_0_20_500 = reg_denoise.predict(features_0_20_500)
denoise_0_20_500 = pd.DataFrame({'stress':pred_0_20_500, 'strain':strain_0_20_500, 'T':T_0_20_500, 'strainrate':strainrate_0_20_500, 'porosity':porosity_0_20_500})

strain_0_20_1000 = []
strain_0_20_1000 = np.arange(0, 0.38, 0.0025)
strainrate_0_20_1000 = []
porosity_0_20_1000 = []
T_0_20_1000 = []
for temp_0_20_1000 in strain_0_20_1000:
    strainrate_0_20_1000.append(1000)
    porosity_0_20_1000.append(0)
    T_0_20_1000.append(20)
features_0_20_1000 = pd.DataFrame({'strain':strain_0_20_1000, 'T':T_0_20_1000, 'strainrate':strainrate_0_20_1000, 'porosity':porosity_0_20_1000})
pred_0_20_1000 = reg_denoise.predict(features_0_20_1000)
denoise_0_20_1000 = pd.DataFrame({'stress':pred_0_20_1000, 'strain':strain_0_20_1000, 'T':T_0_20_1000, 'strainrate':strainrate_0_20_1000, 'porosity':porosity_0_20_1000})

strain_0_20_3000 = []
strain_0_20_3000 = np.arange(0, 0.38, 0.0025)
strainrate_0_20_3000 = []
porosity_0_20_3000 = []
T_0_20_3000 = []
for temp_0_20_3000 in strain_0_20_3000:
    strainrate_0_20_3000.append(3000)
    porosity_0_20_3000.append(0)
    T_0_20_3000.append(20)
features_0_20_3000 = pd.DataFrame({'strain':strain_0_20_3000, 'T':T_0_20_3000, 'strainrate':strainrate_0_20_3000, 'porosity':porosity_0_20_3000})
pred_0_20_3000 = reg_denoise.predict(features_0_20_3000)
denoise_0_20_3000 = pd.DataFrame({'stress':pred_0_20_3000, 'strain':strain_0_20_3000, 'T':T_0_20_3000, 'strainrate':strainrate_0_20_3000, 'porosity':porosity_0_20_3000})

strain_0_20_5000 = []
strain_0_20_5000 = np.arange(0, 0.38, 0.0025)
strainrate_0_20_5000 = []
porosity_0_20_5000 = []
T_0_20_5000 = []
for temp_0_20_5000 in strain_0_20_5000:
    strainrate_0_20_5000.append(5000)
    porosity_0_20_5000.append(0)
    T_0_20_5000.append(20)
features_0_20_5000 = pd.DataFrame({'strain':strain_0_20_5000, 'T':T_0_20_5000, 'strainrate':strainrate_0_20_5000, 'porosity':porosity_0_20_5000})
pred_0_20_5000 = reg_denoise.predict(features_0_20_5000)
denoise_0_20_5000 = pd.DataFrame({'stress':pred_0_20_5000, 'strain':strain_0_20_5000, 'T':T_0_20_5000, 'strainrate':strainrate_0_20_5000, 'porosity':porosity_0_20_5000})

strain_0_200_500 = []
strain_0_200_500 = np.arange(0, 0.38, 0.0025)
strainrate_0_200_500 = []
porosity_0_200_500 = []
T_0_200_500 = []
for temp_0_200_500 in strain_0_200_500:
    strainrate_0_200_500.append(500)
    porosity_0_200_500.append(0)
    T_0_200_500.append(200)
features_0_200_500 = pd.DataFrame({'strain':strain_0_200_500, 'T':T_0_200_500, 'strainrate':strainrate_0_200_500, 'porosity':porosity_0_200_500})
pred_0_200_500 = reg_denoise.predict(features_0_200_500)
denoise_0_200_500 = pd.DataFrame({'stress':pred_0_200_500, 'strain':strain_0_200_500, 'T':T_0_200_500, 'strainrate':strainrate_0_200_500, 'porosity':porosity_0_200_500})

strain_0_200_3000 = []
strain_0_200_3000 = np.arange(0, 0.38, 0.0025)
strainrate_0_200_3000 = []
porosity_0_200_3000 = []
T_0_200_3000 = []
for temp_0_200_3000 in strain_0_200_3000:
    strainrate_0_200_3000.append(3000)
    porosity_0_200_3000.append(0)
    T_0_200_3000.append(200)
features_0_200_3000 = pd.DataFrame({'strain':strain_0_200_3000, 'T':T_0_200_3000, 'strainrate':strainrate_0_200_3000, 'porosity':porosity_0_200_3000})
pred_0_200_3000 = reg_denoise.predict(features_0_200_3000)
denoise_0_200_3000 = pd.DataFrame({'stress':pred_0_200_3000, 'strain':strain_0_200_3000, 'T':T_0_200_3000, 'strainrate':strainrate_0_200_3000, 'porosity':porosity_0_200_3000})

strain_0_400_500 = []
strain_0_400_500 = np.arange(0, 0.38, 0.0025)
strainrate_0_400_500 = []
porosity_0_400_500 = []
T_0_400_500 = []
for temp_0_400_500 in strain_0_400_500:
    strainrate_0_400_500.append(500)
    porosity_0_400_500.append(0)
    T_0_400_500.append(400)
features_0_400_500 = pd.DataFrame({'strain':strain_0_400_500, 'T':T_0_400_500, 'strainrate':strainrate_0_400_500, 'porosity':porosity_0_400_500})
pred_0_400_500 = reg_denoise.predict(features_0_400_500)
denoise_0_400_500 = pd.DataFrame({'stress':pred_0_400_500, 'strain':strain_0_400_500, 'T':T_0_400_500, 'strainrate':strainrate_0_400_500, 'porosity':porosity_0_400_500})

strain_0_400_3000 = []
strain_0_400_3000 = np.arange(0, 0.38, 0.0025)
strainrate_0_400_3000 = []
porosity_0_400_3000 = []
T_0_400_3000 = []
for temp_0_400_3000 in strain_0_400_3000:
    strainrate_0_400_3000.append(3000)
    porosity_0_400_3000.append(0)
    T_0_400_3000.append(400)
features_0_400_3000 = pd.DataFrame({'strain':strain_0_400_3000, 'T':T_0_400_3000, 'strainrate':strainrate_0_400_3000, 'porosity':porosity_0_400_3000})
pred_0_400_3000 = reg_denoise.predict(features_0_400_3000)
denoise_0_400_3000 = pd.DataFrame({'stress':pred_0_400_3000, 'strain':strain_0_400_3000, 'T':T_0_400_3000, 'strainrate':strainrate_0_400_3000, 'porosity':porosity_0_400_3000})

strain_26_25_1200 = []
strain_26_25_1200 = np.arange(0, 0.195, 0.0025)
strainrate_26_25_1200 = []
porosity_26_25_1200 = []
T_26_25_1200 = []
for temp_26_25_1200 in strain_26_25_1200:
    strainrate_26_25_1200.append(1200)
    porosity_26_25_1200.append(26)
    T_26_25_1200.append(25)
features_26_25_1200 = pd.DataFrame({'strain':strain_26_25_1200, 'T':T_26_25_1200, 'strainrate':strainrate_26_25_1200, 'porosity':porosity_26_25_1200})
pred_26_25_1200 = reg_denoise.predict(features_26_25_1200)
denoise_26_25_1200 = pd.DataFrame({'stress':pred_26_25_1200, 'strain':strain_26_25_1200, 'T':T_26_25_1200, 'strainrate':strainrate_26_25_1200, 'porosity':porosity_26_25_1200})

strain_26_25_2300 = []
strain_26_25_2300 = np.arange(0, 0.275, 0.0025)
strainrate_26_25_2300 = []
porosity_26_25_2300 = []
T_26_25_2300 = []
for temp_26_25_2300 in strain_26_25_2300:
    strainrate_26_25_2300.append(2300)
    porosity_26_25_2300.append(26)
    T_26_25_2300.append(25)
features_26_25_2300 = pd.DataFrame({'strain':strain_26_25_2300, 'T':T_26_25_2300, 'strainrate':strainrate_26_25_2300, 'porosity':porosity_26_25_2300})
pred_26_25_2300 = reg_denoise.predict(features_26_25_2300)
denoise_26_25_2300 = pd.DataFrame({'stress':pred_26_25_2300, 'strain':strain_26_25_2300, 'T':T_26_25_2300, 'strainrate':strainrate_26_25_2300, 'porosity':porosity_26_25_2300})

strain_26_25_3600 = []
strain_26_25_3600 = np.arange(0, 0.44, 0.0025)
strainrate_26_25_3600 = []
porosity_26_25_3600 = []
T_26_25_3600 = []
for temp_26_25_3600 in strain_26_25_3600:
    strainrate_26_25_3600.append(3600)
    porosity_26_25_3600.append(26)
    T_26_25_3600.append(25)
features_26_25_3600 = pd.DataFrame({'strain':strain_26_25_3600, 'T':T_26_25_3600, 'strainrate':strainrate_26_25_3600, 'porosity':porosity_26_25_3600})
pred_26_25_3600 = reg_denoise.predict(features_26_25_3600)
denoise_26_25_3600 = pd.DataFrame({'stress':pred_26_25_3600, 'strain':strain_26_25_3600, 'T':T_26_25_3600, 'strainrate':strainrate_26_25_3600, 'porosity':porosity_26_25_3600})

strain_26_25_5200 = []
strain_26_25_5200 = np.arange(0, 0.45, 0.0025)
strainrate_26_25_5200 = []
porosity_26_25_5200 = []
T_26_25_5200 = []
for temp_26_25_5200 in strain_26_25_5200:
    strainrate_26_25_5200.append(5200)
    porosity_26_25_5200.append(26)
    T_26_25_5200.append(25)
features_26_25_5200 = pd.DataFrame({'strain':strain_26_25_5200, 'T':T_26_25_5200, 'strainrate':strainrate_26_25_5200, 'porosity':porosity_26_25_5200})
pred_26_25_5200 = reg_denoise.predict(features_26_25_5200)
denoise_26_25_5200 = pd.DataFrame({'stress':pred_26_25_5200, 'strain':strain_26_25_5200, 'T':T_26_25_5200, 'strainrate':strainrate_26_25_5200, 'porosity':porosity_26_25_5200})

strain_26_100_950 = []
strain_26_100_950 = np.arange(0, 0.115, 0.0025)
strainrate_26_100_950 = []
porosity_26_100_950 = []
T_26_100_950 = []
for temp_26_100_950 in strain_26_100_950:
    strainrate_26_100_950.append(950)
    porosity_26_100_950.append(26)
    T_26_100_950.append(100)
features_26_100_950 = pd.DataFrame({'strain':strain_26_100_950, 'T':T_26_100_950, 'strainrate':strainrate_26_100_950, 'porosity':porosity_26_100_950})
pred_26_100_950 = reg_denoise.predict(features_26_100_950)
denoise_26_100_950 = pd.DataFrame({'stress':pred_26_100_950, 'strain':strain_26_100_950, 'T':T_26_100_950, 'strainrate':strainrate_26_100_950, 'porosity':porosity_26_100_950})

strain_26_100_2200 = []
strain_26_100_2200 = np.arange(0, 0.255, 0.0025)
strainrate_26_100_2200 = []
porosity_26_100_2200 = []
T_26_100_2200 = []
for temp_26_100_2200 in strain_26_100_2200:
    strainrate_26_100_2200.append(2200)
    porosity_26_100_2200.append(26)
    T_26_100_2200.append(100)
features_26_100_2200 = pd.DataFrame({'strain':strain_26_100_2200, 'T':T_26_100_2200, 'strainrate':strainrate_26_100_2200, 'porosity':porosity_26_100_2200})
pred_26_100_2200 = reg_denoise.predict(features_26_100_2200)
denoise_26_100_2200 = pd.DataFrame({'stress':pred_26_100_2200, 'strain':strain_26_100_2200, 'T':T_26_100_2200, 'strainrate':strainrate_26_100_2200, 'porosity':porosity_26_100_2200})

strain_26_100_3000 = []
strain_26_100_3000 = np.arange(0, 0.3475, 0.0025)
strainrate_26_100_3000 = []
porosity_26_100_3000 = []
T_26_100_3000 = []
for temp_26_100_3000 in strain_26_100_3000:
    strainrate_26_100_3000.append(3000)
    porosity_26_100_3000.append(26)
    T_26_100_3000.append(100)
features_26_100_3000 = pd.DataFrame({'strain':strain_26_100_3000, 'T':T_26_100_3000, 'strainrate':strainrate_26_100_3000, 'porosity':porosity_26_100_3000})
pred_26_100_3000 = reg_denoise.predict(features_26_100_3000)
denoise_26_100_3000 = pd.DataFrame({'stress':pred_26_100_3000, 'strain':strain_26_100_3000, 'T':T_26_100_3000, 'strainrate':strainrate_26_100_3000, 'porosity':porosity_26_100_3000})

strain_26_100_4200 = []
strain_26_100_4200 = np.arange(0, 0.456, 0.0025)
strainrate_26_100_4200 = []
porosity_26_100_4200 = []
T_26_100_4200 = []
for temp_26_100_4200 in strain_26_100_4200:
    strainrate_26_100_4200.append(4200)
    porosity_26_100_4200.append(26)
    T_26_100_4200.append(100)
features_26_100_4200 = pd.DataFrame({'strain':strain_26_100_4200, 'T':T_26_100_4200, 'strainrate':strainrate_26_100_4200, 'porosity':porosity_26_100_4200})
pred_26_100_4200 = reg_denoise.predict(features_26_100_4200)
denoise_26_100_4200 = pd.DataFrame({'stress':pred_26_100_4200, 'strain':strain_26_100_4200, 'T':T_26_100_4200, 'strainrate':strainrate_26_100_4200, 'porosity':porosity_26_100_4200})

strain_26_200_1050 = []
strain_26_200_1050 = np.arange(0, 0.115, 0.0025)
strainrate_26_200_1050 = []
porosity_26_200_1050 = []
T_26_200_1050 = []
for temp_26_200_1050 in strain_26_200_1050:
    strainrate_26_200_1050.append(1050)
    porosity_26_200_1050.append(26)
    T_26_200_1050.append(200)
features_26_200_1050 = pd.DataFrame({'strain':strain_26_200_1050, 'T':T_26_200_1050, 'strainrate':strainrate_26_200_1050, 'porosity':porosity_26_200_1050})
pred_26_200_1050 = reg_denoise.predict(features_26_200_1050)
denoise_26_200_1050 = pd.DataFrame({'stress':pred_26_200_1050, 'strain':strain_26_200_1050, 'T':T_26_200_1050, 'strainrate':strainrate_26_200_1050, 'porosity':porosity_26_200_1050})

strain_26_200_1500 = []
strain_26_200_1500 = np.arange(0, 0.17, 0.0025)
strainrate_26_200_1500 = []
porosity_26_200_1500 = []
T_26_200_1500 = []
for temp_26_200_1500 in strain_26_200_1500:
    strainrate_26_200_1500.append(1500)
    porosity_26_200_1500.append(26)
    T_26_200_1500.append(200)
features_26_200_1500 = pd.DataFrame({'strain':strain_26_200_1500, 'T':T_26_200_1500, 'strainrate':strainrate_26_200_1500, 'porosity':porosity_26_200_1500})
pred_26_200_1500 = reg_denoise.predict(features_26_200_1500)
denoise_26_200_1500 = pd.DataFrame({'stress':pred_26_200_1500, 'strain':strain_26_200_1500, 'T':T_26_200_1500, 'strainrate':strainrate_26_200_1500, 'porosity':porosity_26_200_1500})

strain_26_200_1950 = []
strain_26_200_1950 = np.arange(0, 0.206, 0.0025)
strainrate_26_200_1950 = []
porosity_26_200_1950 = []
T_26_200_1950 = []
for temp_26_200_1950 in strain_26_200_1950:
    strainrate_26_200_1950.append(1950)
    porosity_26_200_1950.append(26)
    T_26_200_1950.append(200)
features_26_200_1950 = pd.DataFrame({'strain':strain_26_200_1950, 'T':T_26_200_1950, 'strainrate':strainrate_26_200_1950, 'porosity':porosity_26_200_1950})
pred_26_200_1950 = reg_denoise.predict(features_26_200_1950)
denoise_26_200_1950 = pd.DataFrame({'stress':pred_26_200_1950, 'strain':strain_26_200_1950, 'T':T_26_200_1950, 'strainrate':strainrate_26_200_1950, 'porosity':porosity_26_200_1950})

strain_26_200_2800 = []
strain_26_200_2800 = np.arange(0, 0.33, 0.0025)
strainrate_26_200_2800 = []
porosity_26_200_2800 = []
T_26_200_2800 = []
for temp_26_200_2800 in strain_26_200_2800:
    strainrate_26_200_2800.append(2800)
    porosity_26_200_2800.append(26)
    T_26_200_2800.append(200)
features_26_200_2800 = pd.DataFrame({'strain':strain_26_200_2800, 'T':T_26_200_2800, 'strainrate':strainrate_26_200_2800, 'porosity':porosity_26_200_2800})
pred_26_200_2800 = reg_denoise.predict(features_26_200_2800)
denoise_26_200_2800 = pd.DataFrame({'stress':pred_26_200_2800, 'strain':strain_26_200_2800, 'T':T_26_200_2800, 'strainrate':strainrate_26_200_2800, 'porosity':porosity_26_200_2800})

strain_26_200_3800 = []
strain_26_200_3800 = np.arange(0, 0.26, 0.0025)
strainrate_26_200_3800 = []
porosity_26_200_3800 = []
T_26_200_3800 = []
for temp_26_200_3800 in strain_26_200_3800:
    strainrate_26_200_3800.append(3800)
    porosity_26_200_3800.append(26)
    T_26_200_3800.append(200)
features_26_200_3800 = pd.DataFrame({'strain':strain_26_200_3800, 'T':T_26_200_3800, 'strainrate':strainrate_26_200_3800, 'porosity':porosity_26_200_3800})
pred_26_200_3800 = reg_denoise.predict(features_26_200_3800)
denoise_26_200_3800 = pd.DataFrame({'stress':pred_26_200_3800, 'strain':strain_26_200_3800, 'T':T_26_200_3800, 'strainrate':strainrate_26_200_3800, 'porosity':porosity_26_200_3800})

strain_26_300_1100 = []
strain_26_300_1100 = np.arange(0, 0.135, 0.0025)
strainrate_26_300_1100 = []
porosity_26_300_1100 = []
T_26_300_1100 = []
for temp_26_300_1100 in strain_26_300_1100:
    strainrate_26_300_1100.append(1100)
    porosity_26_300_1100.append(26)
    T_26_300_1100.append(300)
features_26_300_1100 = pd.DataFrame({'strain':strain_26_300_1100, 'T':T_26_300_1100, 'strainrate':strainrate_26_300_1100, 'porosity':porosity_26_300_1100})
pred_26_300_1100 = reg_denoise.predict(features_26_300_1100)
denoise_26_300_1100 = pd.DataFrame({'stress':pred_26_300_1100, 'strain':strain_26_300_1100, 'T':T_26_300_1100, 'strainrate':strainrate_26_300_1100, 'porosity':porosity_26_300_1100})

strain_26_300_1900 = []
strain_26_300_1900 = np.arange(0, 0.235, 0.0025)
strainrate_26_300_1900 = []
porosity_26_300_1900 = []
T_26_300_1900 = []
for temp_26_300_1900 in strain_26_300_1900:
    strainrate_26_300_1900.append(1900)
    porosity_26_300_1900.append(26)
    T_26_300_1900.append(300)
features_26_300_1900 = pd.DataFrame({'strain':strain_26_300_1900, 'T':T_26_300_1900, 'strainrate':strainrate_26_300_1900, 'porosity':porosity_26_300_1900})
pred_26_300_1900 = reg_denoise.predict(features_26_300_1900)
denoise_26_300_1900 = pd.DataFrame({'stress':pred_26_300_1900, 'strain':strain_26_300_1900, 'T':T_26_300_1900, 'strainrate':strainrate_26_300_1900, 'porosity':porosity_26_300_1900})

strain_26_300_2900 = []
strain_26_300_2900 = np.arange(0, 0.345, 0.0025)
strainrate_26_300_2900 = []
porosity_26_300_2900 = []
T_26_300_2900 = []
for temp_26_300_2900 in strain_26_300_2900:
    strainrate_26_300_2900.append(2900)
    porosity_26_300_2900.append(26)
    T_26_300_2900.append(300)
features_26_300_2900 = pd.DataFrame({'strain':strain_26_300_2900, 'T':T_26_300_2900, 'strainrate':strainrate_26_300_2900, 'porosity':porosity_26_300_2900})
pred_26_300_2900 = reg_denoise.predict(features_26_300_2900)
denoise_26_300_2900 = pd.DataFrame({'stress':pred_26_300_2900, 'strain':strain_26_300_2900, 'T':T_26_300_2900, 'strainrate':strainrate_26_300_2900, 'porosity':porosity_26_300_2900})

strain_26_300_3700 = []
strain_26_300_3700 = np.arange(0, 0.335, 0.0025)
strainrate_26_300_3700 = []
porosity_26_300_3700 = []
T_26_300_3700 = []
for temp_26_300_3700 in strain_26_300_3700:
    strainrate_26_300_3700.append(3700)
    porosity_26_300_3700.append(26)
    T_26_300_3700.append(300)
features_26_300_3700 = pd.DataFrame({'strain':strain_26_300_3700, 'T':T_26_300_3700, 'strainrate':strainrate_26_300_3700, 'porosity':porosity_26_300_3700})
pred_26_300_3700 = reg_denoise.predict(features_26_300_3700)
denoise_26_300_3700 = pd.DataFrame({'stress':pred_26_300_3700, 'strain':strain_26_300_3700, 'T':T_26_300_3700, 'strainrate':strainrate_26_300_3700, 'porosity':porosity_26_300_3700})

strain_36_25_1000 = []
strain_36_25_1000 = np.arange(0, 0.236, 0.0025)
strainrate_36_25_1000 = []
porosity_36_25_1000 = []
T_36_25_1000 = []
for temp_36_25_1000 in strain_36_25_1000:
    strainrate_36_25_1000.append(1000)
    porosity_36_25_1000.append(36)
    T_36_25_1000.append(25)
features_36_25_1000 = pd.DataFrame({'strain':strain_36_25_1000, 'T':T_36_25_1000, 'strainrate':strainrate_36_25_1000, 'porosity':porosity_36_25_1000})
pred_36_25_1000 = reg_denoise.predict(features_36_25_1000)
denoise_36_25_1000 = pd.DataFrame({'stress':pred_36_25_1000, 'strain':strain_36_25_1000, 'T':T_36_25_1000, 'strainrate':strainrate_36_25_1000, 'porosity':porosity_36_25_1000})

strain_36_25_2000 = []
strain_36_25_2000 = np.arange(0, 0.279, 0.0025)
strainrate_36_25_2000 = []
porosity_36_25_2000 = []
T_36_25_2000 = []
for temp_36_25_2000 in strain_36_25_2000:
    strainrate_36_25_2000.append(2000)
    porosity_36_25_2000.append(36)
    T_36_25_2000.append(25)
features_36_25_2000 = pd.DataFrame({'strain':strain_36_25_2000, 'T':T_36_25_2000, 'strainrate':strainrate_36_25_2000, 'porosity':porosity_36_25_2000})
pred_36_25_2000 = reg_denoise.predict(features_36_25_2000)
denoise_36_25_2000 = pd.DataFrame({'stress':pred_36_25_2000, 'strain':strain_36_25_2000, 'T':T_36_25_2000, 'strainrate':strainrate_36_25_2000, 'porosity':porosity_36_25_2000})

strain_36_25_3000 = []
strain_36_25_3000 = np.arange(0, 0.45, 0.0025)
strainrate_36_25_3000 = []
porosity_36_25_3000 = []
T_36_25_3000 = []
for temp_36_25_3000 in strain_36_25_3000:
    strainrate_36_25_3000.append(3000)
    porosity_36_25_3000.append(36)
    T_36_25_3000.append(25)
features_36_25_3000 = pd.DataFrame({'strain':strain_36_25_3000, 'T':T_36_25_3000, 'strainrate':strainrate_36_25_3000, 'porosity':porosity_36_25_3000})
pred_36_25_3000 = reg_denoise.predict(features_36_25_3000)
denoise_36_25_3000 = pd.DataFrame({'stress':pred_36_25_3000, 'strain':strain_36_25_3000, 'T':T_36_25_3000, 'strainrate':strainrate_36_25_3000, 'porosity':porosity_36_25_3000})

strain_36_25_3000 = []
strain_36_25_3000 = np.arange(0, 0.45, 0.0025)
strainrate_36_25_3000 = []
porosity_36_25_3000 = []
T_36_25_3000 = []
for temp_36_25_3000 in strain_36_25_3000:
    strainrate_36_25_3000.append(3000)
    porosity_36_25_3000.append(36)
    T_36_25_3000.append(25)
features_36_25_3000 = pd.DataFrame({'strain':strain_36_25_3000, 'T':T_36_25_3000, 'strainrate':strainrate_36_25_3000, 'porosity':porosity_36_25_3000})
pred_36_25_3000 = reg_denoise.predict(features_36_25_3000)
denoise_36_25_3000 = pd.DataFrame({'stress':pred_36_25_3000, 'strain':strain_36_25_3000, 'T':T_36_25_3000, 'strainrate':strainrate_36_25_3000, 'porosity':porosity_36_25_3000})

strain_36_100_1300 = []
strain_36_100_1300 = np.arange(0, 0.15, 0.0025)
strainrate_36_100_1300 = []
porosity_36_100_1300 = []
T_36_100_1300 = []
for temp_36_100_1300 in strain_36_100_1300:
    strainrate_36_100_1300.append(1300)
    porosity_36_100_1300.append(36)
    T_36_100_1300.append(100)
features_36_100_1300 = pd.DataFrame({'strain':strain_36_100_1300, 'T':T_36_100_1300, 'strainrate':strainrate_36_100_1300, 'porosity':porosity_36_100_1300})
pred_36_100_1300 = reg_denoise.predict(features_36_100_1300)
denoise_36_100_1300 = pd.DataFrame({'stress':pred_36_100_1300, 'strain':strain_36_100_1300, 'T':T_36_100_1300, 'strainrate':strainrate_36_100_1300, 'porosity':porosity_36_100_1300})

strain_36_100_2050 = []
strain_36_100_2050 = np.arange(0, 0.228, 0.0025)
strainrate_36_100_2050 = []
porosity_36_100_2050 = []
T_36_100_2050 = []
for temp_36_100_2050 in strain_36_100_2050:
    strainrate_36_100_2050.append(2050)
    porosity_36_100_2050.append(36)
    T_36_100_2050.append(100)
features_36_100_2050 = pd.DataFrame({'strain':strain_36_100_2050, 'T':T_36_100_2050, 'strainrate':strainrate_36_100_2050, 'porosity':porosity_36_100_2050})
pred_36_100_2050 = reg_denoise.predict(features_36_100_2050)
denoise_36_100_2050 = pd.DataFrame({'stress':pred_36_100_2050, 'strain':strain_36_100_2050, 'T':T_36_100_2050, 'strainrate':strainrate_36_100_2050, 'porosity':porosity_36_100_2050})

strain_36_100_2350 = []
strain_36_100_2350 = np.arange(0, 0.275, 0.0025)
strainrate_36_100_2350 = []
porosity_36_100_2350 = []
T_36_100_2350 = []
for temp_36_100_2350 in strain_36_100_2350:
    strainrate_36_100_2350.append(2350)
    porosity_36_100_2350.append(36)
    T_36_100_2350.append(100)
features_36_100_2350 = pd.DataFrame({'strain':strain_36_100_2350, 'T':T_36_100_2350, 'strainrate':strainrate_36_100_2350, 'porosity':porosity_36_100_2350})
pred_36_100_2350 = reg_denoise.predict(features_36_100_2350)
denoise_36_100_2350 = pd.DataFrame({'stress':pred_36_100_2350, 'strain':strain_36_100_2350, 'T':T_36_100_2350, 'strainrate':strainrate_36_100_2350, 'porosity':porosity_36_100_2350})

strain_36_100_3400 = []
strain_36_100_3400 = np.arange(0, 0.366, 0.0025)
strainrate_36_100_3400 = []
porosity_36_100_3400 = []
T_36_100_3400 = []
for temp_36_100_3400 in strain_36_100_3400:
    strainrate_36_100_3400.append(3400)
    porosity_36_100_3400.append(36)
    T_36_100_3400.append(100)
features_36_100_3400 = pd.DataFrame({'strain':strain_36_100_3400, 'T':T_36_100_3400, 'strainrate':strainrate_36_100_3400, 'porosity':porosity_36_100_3400})
pred_36_100_3400 = reg_denoise.predict(features_36_100_3400)
denoise_36_100_3400 = pd.DataFrame({'stress':pred_36_100_3400, 'strain':strain_36_100_3400, 'T':T_36_100_3400, 'strainrate':strainrate_36_100_3400, 'porosity':porosity_36_100_3400})

strain_36_100_3700 = []
strain_36_100_3700 = np.arange(0, 0.408, 0.0025)
strainrate_36_100_3700 = []
porosity_36_100_3700 = []
T_36_100_3700 = []
for temp_36_100_3700 in strain_36_100_3700:
    strainrate_36_100_3700.append(3700)
    porosity_36_100_3700.append(36)
    T_36_100_3700.append(100)
features_36_100_3700 = pd.DataFrame({'strain':strain_36_100_3700, 'T':T_36_100_3700, 'strainrate':strainrate_36_100_3700, 'porosity':porosity_36_100_3700})
pred_36_100_3700 = reg_denoise.predict(features_36_100_3700)
denoise_36_100_3700 = pd.DataFrame({'stress':pred_36_100_3700, 'strain':strain_36_100_3700, 'T':T_36_100_3700, 'strainrate':strainrate_36_100_3700, 'porosity':porosity_36_100_3700})

strain_36_300_1000 = []
strain_36_300_1000 = np.arange(0, 0.113, 0.0025)
strainrate_36_300_1000 = []
porosity_36_300_1000 = []
T_36_300_1000 = []
for temp_36_300_1000 in strain_36_300_1000:
    strainrate_36_300_1000.append(1000)
    porosity_36_300_1000.append(36)
    T_36_300_1000.append(300)
features_36_300_1000 = pd.DataFrame({'strain':strain_36_300_1000, 'T':T_36_300_1000, 'strainrate':strainrate_36_300_1000, 'porosity':porosity_36_300_1000})
pred_36_300_1000 = reg_denoise.predict(features_36_300_1000)
denoise_36_300_1000 = pd.DataFrame({'stress':pred_36_300_1000, 'strain':strain_36_300_1000, 'T':T_36_300_1000, 'strainrate':strainrate_36_300_1000, 'porosity':porosity_36_300_1000})

strain_36_300_2000 = []
strain_36_300_2000 = np.arange(0, 0.2, 0.0025)
strainrate_36_300_2000 = []
porosity_36_300_2000 = []
T_36_300_2000 = []
for temp_36_300_2000 in strain_36_300_2000:
    strainrate_36_300_2000.append(2000)
    porosity_36_300_2000.append(36)
    T_36_300_2000.append(300)
features_36_300_2000 = pd.DataFrame({'strain':strain_36_300_2000, 'T':T_36_300_2000, 'strainrate':strainrate_36_300_2000, 'porosity':porosity_36_300_2000})
pred_36_300_2000 = reg_denoise.predict(features_36_300_2000)
denoise_36_300_2000 = pd.DataFrame({'stress':pred_36_300_2000, 'strain':strain_36_300_2000, 'T':T_36_300_2000, 'strainrate':strainrate_36_300_2000, 'porosity':porosity_36_300_2000})

strain_36_300_3000 = []
strain_36_300_3000 = np.arange(0, 0.357, 0.0025)
strainrate_36_300_3000 = []
porosity_36_300_3000 = []
T_36_300_3000 = []
for temp_36_300_3000 in strain_36_300_3000:
    strainrate_36_300_3000.append(3000)
    porosity_36_300_3000.append(36)
    T_36_300_3000.append(300)
features_36_300_3000 = pd.DataFrame({'strain':strain_36_300_3000, 'T':T_36_300_3000, 'strainrate':strainrate_36_300_3000, 'porosity':porosity_36_300_3000})
pred_36_300_3000 = reg_denoise.predict(features_36_300_3000)
denoise_36_300_3000 = pd.DataFrame({'stress':pred_36_300_3000, 'strain':strain_36_300_3000, 'T':T_36_300_3000, 'strainrate':strainrate_36_300_3000, 'porosity':porosity_36_300_3000})

strain_36_300_4000 = []
strain_36_300_4000 = np.arange(0, 0.436, 0.0025)
strainrate_36_300_4000 = []
porosity_36_300_4000 = []
T_36_300_4000 = []
for temp_36_300_4000 in strain_36_300_4000:
    strainrate_36_300_4000.append(4000)
    porosity_36_300_4000.append(36)
    T_36_300_4000.append(300)
features_36_300_4000 = pd.DataFrame({'strain':strain_36_300_4000, 'T':T_36_300_4000, 'strainrate':strainrate_36_300_4000, 'porosity':porosity_36_300_4000})
pred_36_300_4000 = reg_denoise.predict(features_36_300_4000)
denoise_36_300_4000 = pd.DataFrame({'stress':pred_36_300_4000, 'strain':strain_36_300_4000, 'T':T_36_300_4000, 'strainrate':strainrate_36_300_4000, 'porosity':porosity_36_300_4000})

strain_36_300_4500 = []
strain_36_300_4500 = np.arange(0, 0.5105, 0.0025)
strainrate_36_300_4500 = []
porosity_36_300_4500 = []
T_36_300_4500 = []
for temp_36_300_4500 in strain_36_300_4500:
    strainrate_36_300_4500.append(4500)
    porosity_36_300_4500.append(36)
    T_36_300_4500.append(300)
features_36_300_4500 = pd.DataFrame({'strain':strain_36_300_4500, 'T':T_36_300_4500, 'strainrate':strainrate_36_300_4500, 'porosity':porosity_36_300_4500})
pred_36_300_4500 = reg_denoise.predict(features_36_300_4500)
denoise_36_300_4500 = pd.DataFrame({'stress':pred_36_300_4500, 'strain':strain_36_300_4500, 'T':T_36_300_4500, 'strainrate':strainrate_36_300_4500, 'porosity':porosity_36_300_4500})

# combine all the data after denoising
denoise_data = pd.concat([denoise_0_20_500,denoise_0_20_1000,denoise_0_20_3000,denoise_0_20_5000,denoise_0_200_500,denoise_0_200_3000,\
                          denoise_0_400_500,denoise_0_400_3000,denoise_26_25_1200,denoise_26_25_2300,denoise_26_25_3600,denoise_26_25_5200,\
                          denoise_26_100_950,denoise_26_100_2200,denoise_26_100_3000,denoise_26_100_4200,denoise_26_200_1050,\
                          denoise_26_200_1500,denoise_26_200_1950,denoise_26_200_2800,denoise_26_200_3800,denoise_26_300_1100,\
                          denoise_26_300_1900,denoise_26_300_2900,denoise_26_300_3700,denoise_36_25_1000,denoise_36_25_2000,\
                          denoise_36_25_3000,denoise_36_100_1300,denoise_36_100_2050,denoise_36_100_2350,denoise_36_100_3400,\
                          denoise_36_100_3700,denoise_36_300_1000,denoise_36_300_2000,denoise_36_300_3000,denoise_36_300_4000,\
                          denoise_36_300_4500], ignore_index=True)
