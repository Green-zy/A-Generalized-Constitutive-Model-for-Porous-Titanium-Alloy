# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:20:51 2020

@author: Yun Zhou
"""

def scanning (P,T,S):
    df_path = denoise_data.loc[(denoise_data['porosity']==P)&(denoise_data['T']==T)&(denoise_data['strainrate']==S)]
    return df_path                                                                                       
import pandas as pd
path_1_fitting = pd.DataFrame()
path_2_fitting = pd.DataFrame()
path_3_fitting = pd.DataFrame()
path_4_fitting = pd.DataFrame()
path_5_fitting = pd.DataFrame()
path_6_fitting = pd.DataFrame()
path_7_fitting = pd.DataFrame()
path_8_fitting = pd.DataFrame()

def dr_fitting (Porosity,Temperature,Strainrate):
    adjacent_points(Porosity,Temperature,Strainrate)
    P = Adjacent_porosity
    T = Adjacent_T
    S = Adjacent_strainrate
    path_1_fitting = scanning(P[0],T[0],S[0])
    path_2_fitting = scanning(P[0],T[0],S[1])
    path_3_fitting = scanning(P[0],T[1],S[2])
    path_4_fitting = scanning(P[0],T[1],S[3])
    path_5_fitting = scanning(P[1],T[2],S[4])
    path_6_fitting = scanning(P[1],T[2],S[5])
    path_7_fitting = scanning(P[1],T[3],S[6])
    path_8_fitting = scanning(P[1],T[3],S[7])
    
    list_1 = np.array(path_1_fitting['stress'].tolist())
    temp_len = len(list_1)
    list_2 = np.array(path_2_fitting['stress'].tolist())
    if temp_len > len(list_2):
        temp_len = len(list_2)
    list_3 = np.array(path_3_fitting['stress'].tolist())
    if temp_len > len(list_3):
        temp_len = len(list_3)
    list_4 = np.array(path_4_fitting['stress'].tolist())
    if temp_len > len(list_4):
        temp_len = len(list_4)
    list_5 = np.array(path_5_fitting['stress'].tolist())
    if temp_len > len(list_5):
        temp_len = len(list_5)
    list_6 = np.array(path_6_fitting['stress'].tolist())
    if temp_len > len(list_6):
        temp_len = len(list_6)
    list_7 = np.array(path_7_fitting['stress'].tolist())
    if temp_len > len(list_7):
        temp_len = len(list_7)
    list_8 = np.array(path_8_fitting['stress'].tolist())
    if temp_len > len(list_8):
        temp_len = len(list_8)
    list_1 = list_1[:temp_len]
    list_2 = list_2[:temp_len]
    list_3 = list_3[:temp_len]
    list_4 = list_4[:temp_len]
    list_5 = list_5[:temp_len]
    list_6 = list_6[:temp_len]
    list_7 = list_7[:temp_len]
    list_8 = list_8[:temp_len]

    layer_3_1_f = (Strainrate - S[0])/(S[1] - S[0]) * (list_2 - list_1) + list_1
    layer_3_2_f = (Strainrate - S[2])/(S[3] - S[2]) * (list_4 - list_3) + list_3
    layer_3_3_f = (Strainrate - S[4])/(S[5] - S[4]) * (list_6 - list_5) + list_5
    layer_3_4_f = (Strainrate - S[6])/(S[7] - S[6]) * (list_8 - list_7) + list_7
    
    layer_2_1_f = (Temperature - T[0])/(T[1] - T[0]) * (layer_3_2_f - layer_3_1_f) + layer_3_1_f
    layer_2_2_f = (Temperature - T[2])/(T[3] - T[2]) * (layer_3_4_f - layer_3_3_f) + layer_3_3_f
    
    stress = (Porosity - P[0])/(P[1] - P[0]) * (layer_2_2_f - layer_2_1_f) + layer_2_1_f
    strain = []
    start_strain = 0.0025 * temp_len
    while temp_len !=0 :
        strain.append(start_strain - 0.0025 * temp_len)
        temp_len = temp_len - 1
    stress_strain = pd.DataFrame({'stress':stress, 'strain':strain})
    return stress_strain


# PREDICT = dr_fitting(26,200,3800)
# print(PREDICT)
# final_fig = plt.figure(figsize=(12,9))
# final_plot = final_fig.add_subplot(1, 1, 1)
# plt.plot(PREDICT['strain'], PREDICT['stress'], lw=3, color='orange', label='true stress')
# #plt.plot(final_test_strain, final_pred, lw=3, color='blue',  linestyle=':', label='predicted stress')
# plt.show()
