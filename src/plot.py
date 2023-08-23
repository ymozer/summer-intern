#%%
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sns.set_theme()


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

data = pd.read_csv('/Users/metin/Documents/Intern/T1.csv',sep=';')
print(data.columns)
date_time = pd.to_datetime(data.pop('Date/Time'), format='%d %m %Y %H:%M')

#%%------------ [markdown]-------
plot_cols = ['Wind Speed (m/s)', 'Wind Direction (°)', 'Theoretical_Power_Curve (KWh)']
plot_features = data[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = data[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
# %% draw a graph for showing wind speed and direction
plt.figure(figsize=(12, 6))
plt.plot(data['Wind Speed (m/s)'][:480], label='Wind Speed (m/s)')
plt.plot(data['Wind Direction (°)'][:480], label='Wind Direction (°)')
plt.legend()
plt.show()


# %%
# wind speed and direction circle graph
plt.figure(figsize=(12, 6))
plt.polar(data['Wind Direction (°)'][:480], data['Wind Speed (m/s)'][:480], 'ro')
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.hist2d(data['Wind Direction (°)'], data['Wind Speed (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')


# %% draw a graph for showing wind speed and relation to power
plt.figure(figsize=(12, 6))
plt.plot(data['Wind Speed (m/s)'][:480], label='Wind Speed (m/s)')
plt.plot(data['LV ActivePower (kW)'][:480], label='LV ActivePower (kW)')
plt.legend()
plt.show()


# %%

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('X-axis')
ax1.set_ylabel('kW', color=color)
ax1.plot(data['LV ActivePower (kW)'][:300], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('wspeed', color=color)
ax2.plot(data['Wind Speed (m/s)'][:300], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# direction
ax3 = ax1.twinx()
color = 'tab:green'
ax3.set_ylabel('Dir', color=color)
ax3.plot(data['Wind Direction (°)'][:300], color=color)
ax3.tick_params(axis='y', labelcolor=color)

# theo power
ax4 = ax1.twinx()
color = 'tab:orange'
ax4.set_ylabel('kWh', color=color)
ax4.plot(data['Theoretical_Power_Curve (KWh)'][:300], color=color)
ax4.tick_params(axis='y', labelcolor=color)


fig.tight_layout()
plt.title('Wind Speed (m/s) and LV ActivePower (kW) Relation')
plt.show()


# %%
