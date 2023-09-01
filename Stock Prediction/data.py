import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
desired_directory = 'D:\Internship project\Barath Intern'
os.chdir(desired_directory)




def data():

    AMZN=pd.read_csv('Amazon_stocks.csv')
    TWTR=pd.read_csv('Twitter_stocks.csv')
    TSLA=pd.read_csv('Tesla_stocks.csv')
    AAPL=pd.read_csv('Apple_stock.csv')
    GOOG=pd.read_csv('Google_stocks.csv')

    return AMZN['open'],TWTR['open'],TSLA['open'],AAPL['open'],GOOG['open']

def model():

    AMZN_model=load_model("static\Amazon_Lstm_stock_model.h5")
    TWTR_model =load_model("static\Twitter_Lstm_stock_model.h5")
    TSLA_model =load_model("static\Tesla_Lstm_stock_model.h5")
    AAPL_model =load_model("static\APPl_Lstm_stock_model.h5")
    GOOG_model =load_model("static\Google_Lstm_stock_model.h5")

    return AMZN_model,TWTR_model,TSLA_model,AAPL_model,GOOG_model



def graph(df):
    lnprice=np.log(df)
    plt.plot(lnprice)
    plt.savefig('static/images/Past_plot.jpg')
    
    
    
def Graph(day_new_extended,extended_predictions):
    d_new_lnprice=np.log(day_new_extended)
    ext_pred_lnprice=np.log(extended_predictions)
    
    



