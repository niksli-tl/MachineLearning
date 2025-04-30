import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('car data.csv')
def fill_st(row):
    if row['Seller_Type'] == 'Dealer':
        return 1
    else:
        return 0
df['Seller_Type'] = df.apply(fill_st,axis= 1)
def fill_t(row):
    if row['Transmission'] == 'Manual':
        return 1
    else:
        return 0
df['Transmission'] = df.apply(fill_t,axis= 1)
df[list(pd.get_dummies(df['Fuel_Type']).columns)] = pd.get_dummies(df['Fuel_Type'])
df = df.drop(['Car_Name','Fuel_Type'],axis= 1)
df.info()
x = df.drop('Present_Price', axis= 1)
y = df['Present_Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
score = reg.score(x_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)
print(score)
print(mae)
print(mse)
#0.7266521536479413
#2.642921061007557
#20.35076055684822
