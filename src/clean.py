import datetime
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




df = pd.read_csv('T1.csv', sep=';')
# make all floating number precision to 2
df = df.round(2)
# drop all rows with NaN values
df = df.dropna() 
# drop all rows with 0 values
df = df[(df != 0).all(1)]

# drop Date/Time column
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
df['hour'] = df['Date/Time'].dt.hour
df['minute'] = df['Date/Time'].dt.minute
df.drop(['Date/Time'], axis=1, inplace=True)

X  = df.drop(['LV ActivePower (kW)'], axis=1)
y=df['LV ActivePower (kW)']

X = SelectKBest(f_regression, k=5).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(regressor.score(X_test, y_test))

df.to_csv('T1_cleaned.csv', index=False, sep=';')
