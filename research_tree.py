# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:26:28 2020

@author: Yun Zhou
"""

import numpy as np
layer1 = np.array([0,26,36])
layer2 = np.array([[20,200,400],[25,100,200,300],[25,100,300]])
layer3 = np.array([[[500,1000,3000,5000],[500,3000],[500,3000]],\
          [[1200,2300,3600,5200],[950,2200,3000,4200],\
           [1050,1500,1950,2800,3800],[1100,1900,2900,3700]],\
           [[1000,2000,3000],[1300,2050,2350,3400,3700],\
            [1000,2000,3000,4000,4500]]])

Adjacent_porosity=[]
Adjacent_T=[]
Adjacent_strainrate=[]     
   
def adjacent_points(porosity, T, S):
    #clean original data for adjacent points
     global Adjacent_porosity
     global Adjacent_T
     global Adjacent_strainrate
     Adjacent_porosity.clear()
     Adjacent_T.clear()
     Adjacent_strainrate.clear()
     
     index_layer1 = []
     index_layer2 = []
     index_layer3 = []
     
     # find out adjacent paths
     # for layer 1 
     if porosity < layer1[1]:
          Adjacent_porosity.append(layer1[0])
          Adjacent_porosity.append(layer1[1])
          index_layer1 = [0,1]
     if porosity >= layer1[1]:
          Adjacent_porosity.append(layer1[1])
          Adjacent_porosity.append(layer1[2])
          index_layer1 = [1,2]
    # for layer 2
     count = 0
     for temp_t in layer2[index_layer1[0]]:
         if T > temp_t:
             count = count + 1
     if count == 0:
         index_layer2.append(count)
         index_layer2.append(count+1)
         Adjacent_T.append(layer2[index_layer1[0]][count])
         Adjacent_T.append(layer2[index_layer1[0]][count+1])
     if count == len(layer2[index_layer1[0]]):
         index_layer2.append(count-2)
         index_layer2.append(count-1)
         Adjacent_T.append(layer2[index_layer1[0]][count-2])
         Adjacent_T.append(layer2[index_layer1[0]][count-1])
     if 0 < count < len(layer2[index_layer1[0]]):
         index_layer2.append(count-1)
         index_layer2.append(count)
         Adjacent_T.append(layer2[index_layer1[0]][count-1])
         Adjacent_T.append(layer2[index_layer1[0]][count])  
         
     count = 0    
     for temp_t in layer2[index_layer1[1]]:
         if T > temp_t:
             count = count + 1
     if count == 0:
         index_layer2.append(count)
         index_layer2.append(count+1)
         Adjacent_T.append(layer2[index_layer1[1]][count])
         Adjacent_T.append(layer2[index_layer1[1]][count+1])
     if count == len(layer2[index_layer1[1]]):
         index_layer2.append(count-2)
         index_layer2.append(count-1)
         Adjacent_T.append(layer2[index_layer1[1]][count-2])
         Adjacent_T.append(layer2[index_layer1[1]][count-1])
     if 0 < count < len(layer2[index_layer1[1]]):
         index_layer2.append(count-1)
         index_layer2.append(count)
         Adjacent_T.append(layer2[index_layer1[1]][count-1])
         Adjacent_T.append(layer2[index_layer1[1]][count])
    # for layer3
     count = 0    
     for temp_sr in layer3[index_layer1[0]][index_layer2[0]]:
         if S > temp_sr:
             count = count + 1
     if len(layer3[index_layer1[0]][index_layer2[0]]) == 2:
         index_layer3.append(0)
         index_layer3.append(1)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][0])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][1])
     elif count == 0:
         index_layer3.append(count)
         index_layer3.append(count+1)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][count])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][count+1])
     elif count == len(layer3[index_layer1[0]][index_layer2[0]]):
         index_layer3.append(count-2)
         index_layer3.append(count-1)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][count-2])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][count-1])
     elif 0 < count < len(layer3[index_layer1[0]][index_layer2[0]]):
         index_layer3.append(count-1)
         index_layer3.append(count)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][count-1])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[0]][count])
    
     count = 0    
     for temp_sr in layer3[index_layer1[0]][index_layer2[1]]:
         if S > temp_sr:
             count = count + 1 
     if len(layer3[index_layer1[0]][index_layer2[1]]) == 2:
         index_layer3.append(0)
         index_layer3.append(1)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][0])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][1]) 
     elif count == 0:
         index_layer3.append(count)
         index_layer3.append(count+1)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][count])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][count+1])
     elif count == len(layer3[index_layer1[0]][index_layer2[1]]):
         index_layer3.append(count-2)
         index_layer3.append(count-1)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][count-2])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][count-1])
     elif 0 < count < len(layer3[index_layer1[0]][index_layer2[1]]):
         index_layer3.append(count-1)
         index_layer3.append(count)
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][count-1])
         Adjacent_strainrate.append(layer3[index_layer1[0]][index_layer2[1]][count])
             
     count = 0    
     for temp_sr in layer3[index_layer1[1]][index_layer2[2]]:
         if S > temp_sr:
             count = count + 1 
     if count == 0:
         index_layer3.append(count)
         index_layer3.append(count+1)
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[2]][count])
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[2]][count+1])
     if count == len(layer3[index_layer1[1]][index_layer2[2]]):
         index_layer3.append(count-2)
         index_layer3.append(count-1)
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[2]][count-2])
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[2]][count-1])
     if 0 < count < len(layer3[index_layer1[1]][index_layer2[2]]):
         index_layer3.append(count-1)
         index_layer3.append(count)
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[2]][count-1])
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[2]][count])
           
     count = 0    
     for temp_sr in layer3[index_layer1[1]][index_layer2[3]]:
         if S > temp_sr:
             count = count + 1                
     if count == 0:
         index_layer3.append(count)
         index_layer3.append(count+1)
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[3]][count])
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[3]][count+1])
     if count == len(layer3[index_layer1[1]][index_layer2[3]]):
         index_layer3.append(count-2)
         index_layer3.append(count-1)
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[3]][count-2])
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[3]][count-1])
     if 0 < count < len(layer3[index_layer1[1]][index_layer2[3]]):
         index_layer3.append(count-1)
         index_layer3.append(count)
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[3]][count-1])
         Adjacent_strainrate.append(layer3[index_layer1[1]][index_layer2[3]][count])
     

                 
def adjacent_path(P=[],T=[],S=[]):
    print('Path 1:','R -',P[0],'-',T[0],'-',S[0])
    print('Path 2:','R -',P[0],'-',T[0],'-',S[1])
    print('Path 3:','R -',P[0],'-',T[1],'-',S[2])
    print('Path 4:','R -',P[0],'-',T[1],'-',S[3])
    print('Path 5:','R -',P[1],'-',T[2],'-',S[4])
    print('Path 6:','R -',P[1],'-',T[2],'-',S[5])
    print('Path 7:','R -',P[1],'-',T[3],'-',S[6])
    print('Path 8:','R -',P[1],'-',T[3],'-',S[7])
    

# # An example
# adjacent_points(26,100,2200)
# adjacent_path(Adjacent_porosity, Adjacent_T, Adjacent_strainrate)


