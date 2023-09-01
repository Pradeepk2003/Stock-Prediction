from flask import Flask, render_template,redirect,url_for,request
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import pandas as pd
from datetime import datetime
import os

app=Flask(__name__)

def Preprocess_data(dataset,time_step):
  X,Y=[],[]
  for i in range(len(dataset)-time_step-1):
    a=dataset[i:(i+time_step),0]
    X.append(a)
    b=dataset[i+time_step,0]
    Y.append(b)

  return np.array(X),np.array(Y)


@app.route("/")
def home():
    return "HOME"

@app.route('/Stock_prediction',methods=['POST','GET'])
def stock_prediction():
    com_na = ''
    if request.method == "POST":
        com_na = request.form['company']
        data=''
        model=''
        if com_na=='AAPL':
            csv_path = os.path.join(os.path.dirname(__file__), "Apple_stocks.csv")
            data=pd.read_csv(csv_path)
            model="Apple_Lstm_stock_model.h5"
        elif com_na =='AMZN':
            data=pd.read_csv("Amazon_stocks.csv")
            model="Amazon_Lstm_stock_model.h5"

        elif com_na=='TSLA':
            data=pd.read_csv("Tesla_stocks.csv")
            model="Tesla_Lstm_stock_model.h5"

        elif com_na=='TWTR':
            data=pd.read_csv("Twitter_stocks.csv")
            model="Twitter_Lstm_stock_model.h5"

        elif com_na=='GOOG':
            data=pd.read_csv("Google_stocks.csv")
            model="Google_Lstm_stock_model.h5"
        
        pred_model=load_model(model)
        
        df1=data['open']
        scaler=MinMaxScaler(feature_range=(0,1))
        df=scaler.fit_transform(np.array(df1).reshape(-1,1))
        data_dis=plt.plot(df1)
        training_size=int(len(df)*0.65)
        test_size=len(df)-training_size
        train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]
        x_train,y_train = Preprocess_data(train_data,200)
        x_test,y_test = Preprocess_data(test_data,200)
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
        x_input=test_data[997:].reshape(1,-1)
        x_input.shape
        n_days = 200
        x_input = test_data[-200:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        lst_output = []
        n_steps = 199

        for i in range(n_days):
            if len(temp_input) > 100:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat =pred_model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = pred_model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())

        extended_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        day_new_extended = np.arange(len(df1), len(df1) + n_days)
        plt.plot(df1, label='Original Data',color='blue')
        plt.plot(day_new_extended, extended_predictions, label='Extended Predictions',color='orange')
        plt.savefig("predict.jpg")
        return render_template('index.html', msg=com_na)
    else:
        return render_template('index.html', msg="Please select a company")
       

      

    
    return render_template('index.html',msg=com_na)


if __name__=="__main__":
    app.run(debug=True)