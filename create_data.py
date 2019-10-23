import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# determine two stages of compression process
def ZA_STRAIN_DIVISION(T, strain_ratio, yield_limit):
    elastic_moduli = 195100
    yield_strain = yield_limit / elastic_moduli
    strain_before_yield = np.arange(0, yield_strain, 0.0005)
    strain_after_yield = np.arange(yield_strain, 0.4, 0.0005)
    return strain_before_yield, strain_after_yield

# determine the strain ranges of two stages
def ZA_STRAIN(T, strain_ratio, yield_limit):
    strain_temp =  ZA_STRAIN_DIVISION(T, strain_ratio, yield_limit)
    strain = np.append(strain_temp[0], strain_temp[1])
    return strain

# determine the stress of two stages
def ZA_CONS_MODEL(T, strain_ratio, yield_limit):
    C1 = 167.4473
    C2 = 212.567
    C3 = 0.0238
    C4 = 0.0029
    C5 = 6505.6
    C6 = 1005.2
    n = 1.1435
    B1 = 1.149
    B2 = -4.576*10**(-4)
    B3 = -1.241*10**(-8)
    strain_temp = ZA_STRAIN_DIVISION(T, strain_ratio, yield_limit)
    strain_before_yield = strain_temp[0]
    strain_after_yield = strain_temp[1]
    stress_after_yield = C1 + C2*math.exp(-C3*T + C4*T*math.log(strain_ratio)) + (C5*strain_after_yield**n + C6)*(B1 + B2*T +B3*T**2)
    stress_before_yield =  stress_after_yield[0] / strain_after_yield[0] * strain_before_yield
    stress = np.append(stress_before_yield, stress_after_yield)
    return stress

# set the figure size 
f = plt.figure(figsize=(20,7))

# plot the stress-strain curve(temperature 20℃, strain rate 500/s) and save the data
ax1 = f.add_subplot(2,4,1)
T = 20 + 273.15
strainrate = 500
yieldlimit = 1184.6
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax1.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=20℃, Strain rate=500/s')
T1 = np.array([])
strainrate1 = np.array([])
porosity1 = np.array([])
for temp in strain:
    T1 = np.append(T1, 20)
    strainrate1 = np.append(strainrate1, 500) 
    porosity1 = np.append(porosity1, 0)
data1 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T1, 'strainrate':strainrate1, 'porosity':porosity1})

# plot the stress-strain curve(temperature 200℃, strain rate 500/s) and save the data
ax2 = f.add_subplot(2,4,2)
T = 200 + 273.15
yieldlimit = 1159.6
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax2.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=200℃, Strain rate=500/s')
T2 = np.array([])
strainrate2 = np.array([])
porosity2 = np.array([])
for temp in strain:
    T2 = np.append(T2, 200)
    strainrate2 = np.append(strainrate2, 500) 
    porosity2 = np.append(porosity2, 0)
data2 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T2, 'strainrate':strainrate2, 'porosity':porosity2})

# plot the stress-strain curve(temperature 400℃, strain rate 500/s) and save the data
ax3 = f.add_subplot(2,4,3)
T = 400 + 273.15
yieldlimit = 954.4
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax3.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=400℃, Strain rate=500/s')
T3 = np.array([])
strainrate3 = np.array([])
porosity3 = np.array([])
for temp in strain:
    T3 = np.append(T3, 400)
    strainrate3 = np.append(strainrate3, 500) 
    porosity3 = np.append(porosity3, 0)
data3 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T3, 'strainrate':strainrate3, 'porosity':porosity3})

# plot the stress-strain curve(temperature 20℃, strain rate 1000/s) and save the data
ax4 = f.add_subplot(2,4,4)
T = 20 + 273.15
strainrate = 1000
yieldlimit = 1292.4
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax4.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=20℃, Strain rate=1000/s')
T4 = np.array([])
strainrate4 = np.array([])
porosity4 = np.array([])
for temp in strain:
    T4 = np.append(T4, 20)
    strainrate4 = np.append(strainrate4, 1000) 
    porosity4 = np.append(porosity4, 0)
data4 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T4, 'strainrate':strainrate4, 'porosity':porosity4})

# plot the stress-strain curve(temperature 20℃, strain rate 3000/s) and save the data
ax5 = f.add_subplot(2,4,5)
strainrate = 3000
yieldlimit = 1398.1
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax5.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=20℃, Strain rate=3000/s')
T5 = np.array([])
strainrate5 = np.array([])
porosity5 = np.array([])
for temp in strain:
    T5 = np.append(T5, 20)
    strainrate5 = np.append(strainrate5, 3000) 
    porosity5 = np.append(porosity5, 0)
data5 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T5, 'strainrate':strainrate5, 'porosity':porosity5})

# plot the stress-strain curve(temperature 200℃, strain rate 3000/s) and save the data
ax6 = f.add_subplot(2,4,6)
T = 200 + 273.15
yieldlimit = 1364.1
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax6.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=200℃, Strain rate=3000/s')
T6 = np.array([])
strainrate6 = np.array([])
porosity6 = np.array([])
for temp in strain:
    T6 = np.append(T6, 200)
    strainrate6 = np.append(strainrate6, 3000) 
    porosity6 = np.append(porosity6, 0)
data6 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T6, 'strainrate':strainrate6, 'porosity':porosity6})

# plot the stress-strain curve(temperature 400℃, strain rate 3000/s) and save the data
ax7 = f.add_subplot(2,4,7)
T = 400 + 273.15
yieldlimit = 1149.5
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax7.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=400℃, Strain rate=3000/s')
T7 = np.array([])
strainrate7 = np.array([])
porosity7 = np.array([])
for temp in strain:
    T7 = np.append(T7, 400)
    strainrate7 = np.append(strainrate7, 3000) 
    porosity7 = np.append(porosity7, 0)
data7 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T7, 'strainrate':strainrate7, 'porosity':porosity7})

# plot the stress-strain curve(temperature 20℃, strain rate 5000/s) and save the data
ax8 = f.add_subplot(2,4,8)
T = 20 + 273.15
strainrate = 5000
yieldlimit = 1432.3
strain = ZA_STRAIN(T, strainrate, yieldlimit)
stress = ZA_CONS_MODEL(T, strainrate, yieldlimit)
ax8.plot(strain, stress, 'orange')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('T=20℃, Strain rate=5000/s')
T8 = np.array([])
strainrate8 = np.array([])
porosity8 = np.array([])
for temp in strain:
    T8 = np.append(T8, 20)
    strainrate8 = np.append(strainrate8, 5000) 
    porosity8 = np.append(porosity8, 0)
data8 = pd.DataFrame({'stress':stress, 'strain':strain, 'T':T8, 'strainrate':strainrate8, 'porosity':porosity8})

# adjust the space between subplots
plt.subplots_adjust(wspace=0.25, hspace=0.4)
plt.show()

# combine these 8 group data to one dataframe
data_supplement = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8], ignore_index=True)