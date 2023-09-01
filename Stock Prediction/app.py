from flask import Flask, render_template,redirect,url_for,request
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import pandas as pd
from datetime import datetime
import os
import data as grp
AMZN,TWTR,TSLA,AAPL,GOOG=grp.data()
AMZN_model,TWTR_model,TSLA_model,AAPL_model,GOOG_model=grp.model()
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
            data=AAPL
            model=AAPL_model

        elif com_na =='AMZN':
            data=AMZN
            model=AMZN_model

        elif com_na=='TSLA':
            data= TSLA
            model=TSLA_model

        elif com_na=='TWTR':
            data=TWTR
            model=TWTR_model

        elif com_na=='GOOG':
            data=GOOG
            model=GOOG_model


         
        df1=list(data)
        scaler=MinMaxScaler(feature_range=(0,1))
        df=scaler.fit_transform(np.array(df1).reshape(-1,1))
        training_size=int(len(df)*0.65)
        test_size=len(df)-training_size
        train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]
        x_test,y_test = Preprocess_data(test_data,200)
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
        n_days = 200
        x_input = test_data[-200:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        lst_output = []
        n_steps = 200

        for i in range(n_days):
            if len(temp_input) > 100:
                x_input = np.array(temp_input[0:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())

        extended_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        day_new_extended = np.arange(len(df1), len(df1) + n_days)
        plt.plot(df1)
        plt.savefig('plot_plot.jpg')
        plt.plot(day_new_extended,extended_predictions, label="Test Prediction", color="green")
        plt.savefig('predicted_plot.jpg')
        return lst_output

    else:
        return render_template('index.html', msg="Please select a company")
    return render_template('index.html',msg=com_na)
@app.route('/Prediction')
def plotout():
      return render_template('prediction.html')

if __name__=="__main__":
    app.run(debug=True)