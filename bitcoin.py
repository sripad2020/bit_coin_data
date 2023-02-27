import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('BTC_USD_Price_Prediction_Data.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
values=data.columns.values
for i in values:
    Nanvalues=data[f'{i}'].isna().sum()
    info=data[f'{i}'].info()
    description=data[f'{i}'].describe()
    print(Nanvalues)
    print(info)
    print(description)
currency=data.Currency.unique()
print(currency)
import seaborn as sn
sn.heatmap(data.corr())
plt.show()
sn.pairplot(data)
plt.show()
plt.plot(data['Closing Price (USD)'],color='red')
plt.plot(data['24h Open (USD)'],color='green')
plt.plot(data['24h High (USD)'],color='yellow')
plt.plot(data['24h Low (USD)'],color='blue')
plt.xlabel('red-<=Closing_price,green-<=opening price,mint-<=High Price,black-<=low price')
plt.legend()
plt.show()

sn.boxplot(data['24h Open (USD)'])
plt.show()
sn.boxplot(data['24h High (USD)'])
plt.show()
sn.boxplot(data['24h Low (USD)'])
plt.show()
sn.boxplot(data['Closing Price (USD)'])
plt.show()

q1=data['24h Open (USD)'].quantile(0.25)
q3=data['24h Open (USD)'].quantile(0.75)

qua1=data['24h High (USD)'].quantile(0.25)
qua3=data['24h High (USD)'].quantile(0.75)

quant_1=data['24h Low (USD)'].quantile(0.25)
quant_3=data['24h Low (USD)'].quantile(0.75)

quantile1=data['Closing Price (USD)'].quantile(0.25)
quantile3=data['Closing Price (USD)'].quantile(0.75)

iqr=q3-q1
iqR=qua3-qua1
Iqr=quant_3-quant_1
IQR=quantile3-quantile1

u_l=q3+1.5*iqr
l_l=q1-1.5*iqr

up_lim=qua3+1.5*iqR
lo_l=qua1-1.5*iqR

upp_lim=quant_3+1.5*Iqr
low_lim=quant_1-1.5*Iqr

upper_limit=quantile3+1.5*IQR
lower_limit=quantile1-1.5*IQR

df=data[(data['24h Open (USD)'] <= u_l) | (data['24h Open (USD)'] >= l_l) | (data['24h High (USD)'] <=up_lim) |(data['24h High (USD)'] >= low_lim)|(data['24h Low (USD)'] <=upp_lim) | (data['24h Low (USD)'] >=low_lim) |(data['Closing Price (USD)'] <= upper_limit)|(data['Closing Price (USD)'] >= lower_limit)]
print(data.shape)
print(df.shape)

q1=df['24h Open (USD)'].quantile(0.25)
q3=df['24h Open (USD)'].quantile(0.75)

qua1=df['24h High (USD)'].quantile(0.25)
qua3=df['24h High (USD)'].quantile(0.75)

quant_1=df['24h Low (USD)'].quantile(0.25)
quant_3=data['24h Low (USD)'].quantile(0.75)

quantile1=df['Closing Price (USD)'].quantile(0.25)
quantile3=df['Closing Price (USD)'].quantile(0.75)

iqr=q3-q1
iqR=qua3-qua1
Iqr=quant_3-quant_1
IQR=quantile3-quantile1

u_l=q3+1.5*iqr
l_l=q1-1.5*iqr

up_lim=qua3+1.5*iqR
lowe_l=qua1-1.5*iqR

upp_lim=quant_3+1.5*Iqr
low_lim=quant_1-1.5*Iqr

upper_limit=quantile3+1.5*IQR
lower_limit=quantile1-1.5*IQR

new_df=df[(df['24h Open (USD)'] <= u_l) & (df['24h Open (USD)'] >= l_l) & (df['24h High (USD)'] <=up_lim) &(df['24h High (USD)'] >= lowe_l)&(df['24h Low (USD)'] <=upp_lim) & (df['24h Low (USD)'] >=low_lim) &(df['Closing Price (USD)'] <= upper_limit)&(df['Closing Price (USD)'] >= lower_limit)]
print(new_df.shape)
print(new_df)

sn.boxplot(new_df['24h Open (USD)'])
plt.show()
sn.boxplot(new_df['24h High (USD)'])
plt.show()
sn.boxplot(new_df['24h Low (USD)'])
plt.show()
sn.boxplot(new_df['Closing Price (USD)'])
plt.show()

from sklearn.model_selection import train_test_split
#x=data[['24h Open (USD)','Closing Price (USD)']]
x_train,x_test,y_train,y_test=train_test_split(new_df[['24h Open (USD)','24h High (USD)']],new_df['Closing Price (USD)'])
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses
model=Sequential()
model.add(Dense(units=new_df[['24h Open (USD)','24h High (USD)']].shape[1],input_dim=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(units=new_df[['24h Open (USD)','24h High (USD)']].shape[1],activation=keras.activations.relu))
model.add(Dense(units=new_df[['24h Open (USD)','24h High (USD)']].shape[1],activation=keras.activations.relu))
model.add(Dense(units=new_df[['24h Open (USD)','24h High (USD)']].shape[1],activation=keras.activations.relu))
model.add(Dense(units=new_df[['24h Open (USD)','24h High (USD)']].shape[1],activation=keras.activations.relu))
model.add(Dense(units=1,activation=keras.activations.relu))
model.compile(optimizer='adam',loss=keras.losses.mean_absolute_error,metrics='mae')
model.fit(x_train,y_train,batch_size=20,epochs=30)
pred=model.predict(x_test)
from sklearn.metrics import r2_score
val=r2_score(y_test,pred)
print(val)
from sklearn.metrics import mean_squared_error
import numpy as np
mse=mean_squared_error(y_test,pred)
print(mse)
print(np.sqrt(mse))