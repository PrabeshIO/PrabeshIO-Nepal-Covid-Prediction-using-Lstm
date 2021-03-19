import numpy as np
from numpy import array
import pandas as pd
from dataframes import province as pr, timeline as t1
from sklearn.preprocessing import MinMaxScaler
import os
import math
import statistics
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

def model():
        df, last_date = t1.get_timeline()
        # print(b)
        df=df.drop(['total_deaths','new_deaths','iso_code','continent'],axis=1)
        df=df[45:] #removing datas where there is no new cases 
        df=df.reset_index()

        df1=df['new_cases']
        scaler=MinMaxScaler()
        df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

        ##splitting dataset into train and test split
        training_size=int(len(df1)*0.75)
        test_size=len(df1)-training_size
        train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = 3
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        # print(y_test)

        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        # X_test.shape

        model=Sequential()
        model.add(LSTM(128,activation='relu',use_bias=True, bias_initializer='ones',return_sequences=True,input_shape=(time_step,1)))
        model.add(Dropout(0.2))
        model.add(LSTM(64,return_sequences=True,activation='relu'))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error',optimizer=opt)

        model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,steps_per_epoch=26,batch_size=16,verbose=1)
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        ##Transformback to original form
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        y_train=scaler.inverse_transform(y_train.reshape(-1,1))
        y_test=scaler.inverse_transform(y_test.reshape(-1,1))

        #plottinng total data and predicted data
        alll=scaler.inverse_transform(df1).tolist()
        predicted= train_predict.tolist()+ test_predict.tolist()
        plot_graph('total_plot',alll[80:],predicted[80:])

        #plotting the training prediction graph
        plot_graph('train_plot',y_train[80:],train_predict[80:])
        
        
        #plotting test prediction graph
        plot_graph('test_plot',y_test,test_predict)


        # prediction for future data
        x_input=test_data[len(test_data)-3:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        lst_output=[]
        n_steps=3
        i=0
        while(i<1):
        
                if(len(temp_input)>3):
                        x_input=np.array(temp_input[1:])
                        x_input=x_input.reshape(1,-1)
                        x_input = x_input.reshape((1, n_steps, 1))
                        yhat = model.predict(x_input, verbose=0)
                        temp_input.extend(yhat[0].tolist())
                        temp_input=temp_input[1:]
                        lst_output.extend(yhat.tolist())
                        i=i+1
                else:
                        x_input = x_input.reshape((1, n_steps,1))
                        yhat = model.predict(x_input, verbose=0)
                        temp_input.extend(yhat[0].tolist())
                        lst_output.extend(yhat.tolist())
                        i=i+1
                

        prev= df['new_cases'][len(df)-5:].tolist()
        new=scaler.inverse_transform(np.array(lst_output[0][0]).reshape(-1,1))
        new=math.floor(new[0][0])
        data=alll[:len(predicted)]
        
        print("RMSE:", math.sqrt(mean_squared_error(data, predicted)))
        print("train mse:", math.sqrt(mean_squared_error(y_train, train_predict)))
        print("test mse:", math.sqrt(mean_squared_error(y_test, test_predict)))
        return(prev,new,math.sqrt(mean_squared_error(data, predicted)))

def plot_graph(name,data,pred):
        plt.clf()
        plt.plot(data,color='green',label='actual data')
        plt.plot(pred,color='red',label='predicted data')
        plt.xlabel('Time')
        plt.ylabel('Cases')
        plt.title(name+" vs Predicted data")
        plt.legend()
        plt.savefig('static/dist/img/'+name+'.png')