#%%
import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sns.set_theme()



listdir = os.listdir('models')
listdir = [file for file in listdir if file.endswith('.txt')]
listdir.sort()

metrics = []
for file in listdir:
    with open('models/'+file, 'r') as f:
        common_ratio  = file.split('_')[2]
        unique_ratio  = file.split('_')[3]
        common_unique = f"{common_ratio}_{unique_ratio}"
    
        breakdown = f.read().split('\n')
        second_line = breakdown[1].split(' ')

        # algo name
        second_line[0] = second_line[0].split('(')[0]
        algo = second_line[0]
        if algo == "XGBRegressor":
            a = breakdown[11].split(')')[1].strip().split(' ')
            # add XGBRegressor as 1st element and
            a.insert(0, algo)
            a.insert(0, common_unique)
            metrics.append(a)
        else:
            second_line.insert(0, common_unique)
            metrics.append(second_line)
LGBM = []
XGB = []
RDF = []
DTR = []

for i in metrics:
    common_unique = i[0]
    algo = i[1]
    mse = i[3]
    mae = i[5]
    r2 = i[7]
    #print(f"{common_unique} {algo} MSE: {mse} MAE: {mae} R2: {r2}")
    if algo == "LGBMRegressor":
        lgbm= []
        lgbm.append(i[0])
        lgbm.append(i[3])
        lgbm.append(i[5])
        lgbm.append(i[7])
        LGBM.append(lgbm)
    if algo == "XGBRegressor":
        xgb = []
        xgb.append(i[0])
        xgb.append(i[3])
        xgb.append(i[5])
        xgb.append(i[7])
        XGB.append(xgb)
    if algo == "RandomForestRegressor":
        rdf = []
        rdf.append(i[0])
        rdf.append(i[3])
        rdf.append(i[5])
        rdf.append(i[7])
        RDF.append(rdf)
    if algo == "DecisionTreeRegressor":
        dtr = []
        dtr.append(i[0])
        dtr.append(i[3])
        dtr.append(i[5])
        dtr.append(i[7])
        DTR.append(dtr)

# plot metrics by algo
#%%
mse=[i[1] for i in LGBM]
# remove the last comma and convert to float
mse = [round(float(i[:-1]),2) for i in mse]
mae=[i[2] for i in LGBM]
mae = [round(float(i[:-1]),2) for i in mae]
r2=[i[3] for i in LGBM]
r2 = [round(float(i[:-1]),2) for i in r2]
x = [i[0] for i in LGBM]

# make 3 subplots for mse, mae, r2 in different colors
fig, ax = plt.subplots(3, 1, figsize=(12, 6))
ax[0].plot(x, mse, 'r')
ax[0].set_title('LGBMRegressor')
ax[0].set_ylabel('MSE')
ax[0].set_xlabel('Common_Unique Ratio')
ax[1].plot(x, mae, 'b')
ax[1].set_ylabel('MAE')
ax[2].plot(x, r2, 'g')
ax[2].set_ylabel('R2')
plt.show()
#%%
mse=[i[1] for i in XGB]
# remove the last comma and convert to float
mse = [round(float(i[:-1]),2) for i in mse]
mae=[i[2] for i in XGB]
mae = [round(float(i[:-1]),2) for i in mae]
r2=[i[3] for i in XGB]
r2 = [round(float(i[:-1]),2) for i in r2]
x = [i[0] for i in XGB]

# make 3 subplots for mse, mae, r2 in different colors
fig, ax = plt.subplots(3, 1, figsize=(12, 6))
ax[0].plot(x, mse, 'r')
ax[0].set_title('XGBRegressor')
ax[0].set_ylabel('MSE')
ax[0].set_xlabel('Common_Unique Ratio')
ax[1].plot(x, mae, 'b')
ax[1].set_ylabel('MAE')
ax[2].plot(x, r2, 'g')
ax[2].set_ylabel('R2')
plt.show()
#%%
mse=[i[1] for i in RDF]
# remove the last comma and convert to float
mse = [round(float(i[:-1]),2) for i in mse]
mae=[i[2] for i in RDF]
mae = [round(float(i[:-1]),2) for i in mae]
r2=[i[3] for i in RDF]
r2 = [round(float(i[:-1]),2) for i in r2]
x = [i[0] for i in RDF]

# make 3 subplots for mse, mae, r2 in different colors
fig, ax = plt.subplots(3, 1, figsize=(12, 6))
ax[0].plot(x, mse, 'r')
ax[0].set_title('RandomForestRegressor')
ax[0].set_ylabel('MSE')
ax[0].set_xlabel('Common_Unique Ratio')
ax[1].plot(x, mae, 'b')
ax[1].set_ylabel('MAE')
ax[2].plot(x, r2, 'g')
ax[2].set_ylabel('R2')
plt.show()
#%%
mse=[i[1] for i in DTR]
# remove the last comma and convert to float
mse = [round(float(i[:-1]),2) for i in mse]
mae=[i[2] for i in DTR]
mae = [round(float(i[:-1]),2) for i in mae]
r2=[i[3] for i in DTR]
r2 = [round(float(i[:-1]),2) for i in r2]
x = [i[0] for i in DTR]

# make 3 subplots for mse, mae, r2 in different colors
fig, ax = plt.subplots(3, 1, figsize=(12, 6))
ax[0].plot(x, mse, 'r')
ax[0].set_title('DecisionTreeRegressor')
ax[0].set_ylabel('MSE')
ax[0].set_xlabel('Common_Unique Ratio')
ax[1].plot(x, mae, 'b')
ax[1].set_ylabel('MAE')
ax[2].plot(x, r2, 'g')
ax[2].set_ylabel('R2')
plt.show()
#%%










    
    
    