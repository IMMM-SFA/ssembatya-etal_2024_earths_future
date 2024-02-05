# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:38:08 2022

@author: hssemba
"""
import sys
#save all outputs into txt file
output_file = open("model_performance_v25.txt", "w")
sys.stdout = output_file
import time
# get the start time
st = time.time()


'''Importing variables'''
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

#%matplotlib inline
from sklearn.ensemble import RandomForestRegressor 

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.neural_network import MLPRegressor as MLP
import datetime
from datetime import datetime, timedelta
import glob


''' Variable Definition 
T2 -> Temperature at 2m in Kelvins 
Q2 ->Water vapor mixing ratio at 2m in kgkg-1 
SWDOWN ->Downward shortwave flux at ground surface in Wm-2 
GLW ->Downward longwave flux at ground surface in Wm-2
WSPD -> WSPD Wind speed ms-1 
T3 -> Temperature Celcius
'''

'''Importing Data'''

#Import 2016 WRF
df_2016x=pd.read_csv("ERCO_WRF_Hourly_Mean_Meteorology_2016.csv")

#Import 2017 WRF
df_2017x=pd.read_csv("ERCO_WRF_Hourly_Mean_Meteorology_2017.csv")

#Import 2016 Residential dd base case
df_ddbase=pd.read_csv("ercot_2016_hourly_consumption_base.csv")

#Import 2016 Residential dd high efficiency
df_ddhi=pd.read_csv("ercot_2016_hourly_consumption_high_efficiency.csv")

#Import 2016 Residential dd standard efficiency
df_ddstandd=pd.read_csv("ercot_2016_hourly_consumption_standard_efficiency.csv")

#Import 2016 Residential dd ultra high efficiency
df_ddultra=pd.read_csv("ercot_2016_hourly_consumption_ultra_high_efficiency.csv")

#Import non-residential load
df_nonres=pd.read_csv("non_res_load_16x.csv")


'''Normalize '''


'''Manipulating df_2016 inputs'''
#combining the 2016 
df_2016=pd.concat([df_2016x, df_2017x]).reset_index()
df_2016=df_2016.drop(["index"], axis=1)

#load in historical scenarios from kwh to mwh
df_ddbase["ResLoad_mwh"]=df_ddbase["total_site_electricity_kwh"]/1000
df_ddstandd["ResLoad_mwh"]=df_ddstandd["total_site_electricity_kwh"]/1000
df_ddhi["ResLoad_mwh"]=df_ddhi["total_site_electricity_kwh"]/1000
df_ddultra["ResLoad_mwh"]=df_ddultra["total_site_electricity_kwh"]/1000

#.........................................................................................
def days_actual(x):
    if x==6:
        return "Sunday"   
    elif x==0:
        return "Monday"  
    elif x==1:
        return "Tuesday"                     
    elif x==2:
        return "Wednesday"
    elif x==3:
        return "Thursday"
    elif x==4:
        return "Friday"
    elif x==5:
        return "Saturday"
        
def one_hot(df_new):

    # One hot encoding certain categorical features
    cat_vars=[ "Hour"]#,"Month"]
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df_new[var], prefix=var)
        data1=df_new.join(cat_list)
        df_new=data1

    data_vars=df_new.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]
    df_new=df_new[to_keep]
    
    return df_new

def weekend(x):
    #5 is Saturday
    if x==5:
        return 1
    #6 is Sunday
    elif x==6:
        return 1  
    else:
        return 0     
#..............................................................................................



#preprocessing, converting time to CT for 2016 and future years
def pre_process(df_new,df_new_plus1,my_year):
    my_date_index = df_new['Time_UTC']
    '''1. Convert time from UTC to CT '''
    df_new['Time_UTC']= pd.to_datetime(df_new['Time_UTC'], errors='coerce')
    #the reason we are subtracting 5 instead of 6 is because the target variable starts from 1 (we want time from 1:00am)
    df_new["Time_CT"]=df_new["Time_UTC"]-timedelta(hours=5)
    
    if df_new.equals(df_2016):
        df_new=df_new[6:8790].reset_index()
        df_new=df_new.drop(["index"], axis=1)
        del df_new["Time_UTC"]
    else:
          
        if my_year !=2099:
        
            df_new=pd.concat([df_new[5:],df_new_plus1[:5]],ignore_index=True)
            df_new["Time_CT"] = my_date_index
            del df_new["Time_UTC"]
        
        else:
            df_new=df_new[5:].reset_index(drop=True)
            del df_new["Time_UTC"]
            
        

    
    '''2. Kelvin to Celius '''
    df_new["T3"]=df_new["T2"]- 273.15
    
    
    '''Unit Conversions '''
    df_new['Time_CT']= pd.to_datetime(df_new['Time_CT'], errors='coerce')
    df_new['Hour'] = pd.DatetimeIndex(df_new['Time_CT']).hour
    #df_new['Month'] = pd.DatetimeIndex(df_new['Time_CT']).month
    df_new['Day_of_week']=df_new['Time_CT'].apply(lambda time: time.dayofweek)
    df_new["Weekend"]=df_new['Day_of_week'].apply(weekend)
    
    #Dropping variables not needed
    df_new=df_new.drop(["T2", "Time_CT", "Day_of_week"], axis=1)

    
    #onehot encoding the categorical variables
    df_new=one_hot(df_new)

    return df_new

    
'''---Metrics (r-squared, MAE, MAPE)---'''
def metrics_test_data(y_test, model_y_pred):
    
    r2= sklearn.metrics.r2_score(y_test, model_y_pred)
    print("R-squared:" ,r2)
    
    mae= sklearn.metrics.mean_absolute_error(y_test, model_y_pred)
    print("Mean absolute error:" ,mae)
    
    y_test, model_y_pred = np.array(y_test), np.array(model_y_pred)
    mape = np.mean(np.abs((y_test - model_y_pred)/y_test))*100
    print("Mean absolute Percentage error:" ,mape)       


'''Training the model'''
#Pre-processing the 2016 data
df_2016_mod=pre_process(df_2016,df_2016,2016)


#Divide into training and testing

# remove variable "Weekend" for X for base, standard, high, ultra 
X=df_2016_mod.drop(["Weekend"], axis=1)

#base case
y_b=df_ddbase[['ResLoad_mwh']]


#High efficiency
y_h=df_ddhi[['ResLoad_mwh']]

# Standard efficiency
y_s=df_ddstandd[['ResLoad_mwh']]

# Ultra high efficiency
y_u=df_ddultra[['ResLoad_mwh']]


# include variable "Weekend" for nonres
X_nonres=df_2016_mod

#non-residential load
y_nonres=df_nonres[["NonResLoad_mwh"]]



# '''....................................................... '''
#splitting the data

#base case
X_trainb, X_testb, y_trainb, y_testb = train_test_split(X, y_b, train_size= 0.7, test_size=0.3, random_state=123)
#normalize x
normb =MinMaxScaler().fit(X_trainb)
X_train_normb=normb.transform(X_trainb)
X_test_normb = normb.transform(X_testb)
#normalize y
ynormb=MinMaxScaler().fit(y_trainb)
y_train_normb=ynormb.transform(y_trainb)


# Standard efficiency
X_trains, X_tests, y_trains, y_tests = train_test_split(X, y_s, train_size= 0.7, test_size=0.3, random_state=123)
#normalize x
norms =MinMaxScaler().fit(X_trains)
X_train_norms=norms.transform(X_trains)
X_test_norms = norms.transform(X_tests)
#normalize y
ynorms=MinMaxScaler().fit(y_trains)
y_train_norms=ynorms.transform(y_trains)


#High efficiency
X_trainh, X_testh, y_trainh, y_testh = train_test_split(X, y_h, train_size= 0.7, test_size=0.3, random_state=123)
#normalize x
normh =MinMaxScaler().fit(X_trainh)
X_train_normh=normh.transform(X_trainh)
X_test_normh = normh.transform(X_testh)
#normalize y
ynormh=MinMaxScaler().fit(y_trainh)
y_train_normh=ynormh.transform(y_trainh)


#Ultra high efficiency
X_trainu, X_testu, y_trainu, y_testu = train_test_split(X, y_u, train_size= 0.7, test_size=0.3, random_state=123)
#normalize x
normu =MinMaxScaler().fit(X_trainu)
X_train_normu=normu.transform(X_trainu)
X_test_normu = normu.transform(X_testu)
#normalize y
ynormu=MinMaxScaler().fit(y_trainu)
y_train_normu=ynormu.transform(y_trainu)


#Non_residential Load
X_trainn, X_testn, y_trainn, y_testn = train_test_split(X_nonres , y_nonres, train_size= 0.7, test_size=0.3, random_state=123)
#normalize x
normn =MinMaxScaler().fit(X_trainn)
X_train_normn=normn.transform(X_trainn)
X_test_normn = normn.transform(X_testn)
#normalize y
ynormn=MinMaxScaler().fit(y_trainn)
y_train_normn=ynormn.transform(y_trainn)

#.......................................................................................................................................................


'''.........................................................'''                 
#Training the model and testing using cross validation

#base case
MLP_modelb = MLP(activation= 'relu', hidden_layer_sizes= (1452,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)
                                      

MLP_scoreb = cross_val_score(MLP_modelb, X_train_normb, y_train_normb, cv=5, scoring='r2')

print("CV scores for base case : ", MLP_scoreb)
print('Mean CV score for base case: {:.3f}'.format(MLP_scoreb.mean()))


#..................................................................................................................
# Standard efficiency
MLP_models = MLP(activation= 'relu', hidden_layer_sizes= (1830,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)

MLP_scores = cross_val_score(MLP_models, X_train_norms, y_train_norms, cv=5, scoring='r2')

print("CV scores for Standard Efficiency : ", MLP_scores)
print('Mean CV score for Standard Efficiency: {:.3f}'.format(MLP_scores.mean()))

#..................................................................................................................
#High efficiency
MLP_modelh = MLP(activation= 'relu', hidden_layer_sizes= (1691,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)

MLP_scoreh = cross_val_score(MLP_modelh, X_train_normh ,y_train_normh, cv=5,  scoring='r2')

print("CV scores for High Efficiency : ", MLP_scoreh)
print('Mean CV score for High Efficiency: {:.3f}'.format(MLP_scoreh.mean()))



#..................................................................................................................
# Ultra high efficiency
MLP_modelu = MLP(activation= 'relu', hidden_layer_sizes= (1691,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)

MLP_scoreu = cross_val_score(MLP_modelu, X_train_normu,y_train_normu, cv=5, scoring='r2')

print("CV scores for Ultra Efficiency : ", MLP_scoreu)
print('Mean CV score for Ultra Efficiency: {:.3f}'.format(MLP_scoreu.mean()))

#..................................................................................................................
# Non-residential Load
MLP_modeln = MLP(activation= 'relu', hidden_layer_sizes= (1223,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)

MLP_scoren = cross_val_score(MLP_modeln, X_train_normn,y_train_normn, cv=5, scoring='r2')

print("CV scores for Non Residential : ", MLP_scoren)
print('Mean CV score for Non Residential: {:.3f}'.format(MLP_scoren.mean()))



'''.................................................'''
#Fitting the model with the training data and predicting with test data
#..................................................................................................................
#base case
MLP_modelb.fit(X_train_normb,y_train_normb)

'''Predicting the load using test data'''
MLP_y_pred_normb=MLP_modelb.predict(X_test_normb).reshape(-1,1)
MLP_y_predb = ynormb.inverse_transform(MLP_y_pred_normb)

#..................................................................................................................
# Standard efficiency
MLP_models.fit(X_train_norms,y_train_norms)

'''Predicting the load using test data'''
MLP_y_pred_norms=MLP_models.predict(X_test_norms).reshape(-1,1)
MLP_y_preds = ynorms.inverse_transform(MLP_y_pred_norms)


#..................................................................................................................
#High efficiency
MLP_modelh.fit(X_train_normh,y_train_normh)

'''Predicting the load using test data'''
MLP_y_pred_normh=MLP_modelh.predict(X_test_normh).reshape(-1,1)
MLP_y_predh = ynormh.inverse_transform(MLP_y_pred_normh)


#..................................................................................................................
# Ultra high efficiency
MLP_modelu.fit(X_train_normu,y_train_normu)

'''Predicting the load using test data'''
MLP_y_pred_normu=MLP_modelu.predict(X_test_normu).reshape(-1,1)
MLP_y_predu = ynormu.inverse_transform(MLP_y_pred_normu)
#..................................................................................................................

#Non-residential load
MLP_modeln.fit(X_train_normn,y_train_normn)

'''Predicting the load using test data'''
MLP_y_pred_normn=MLP_modeln.predict(X_test_normn).reshape(-1,1)
MLP_y_predn = ynormn.inverse_transform(MLP_y_pred_normn)

'''...........................................................'''

#Metrics on Prediction
print("...................Metrics on test data..................")

#Base case
print("........Base Case Prediction.....")
results_base=metrics_test_data(y_testb, MLP_y_predb)

#Standard efficiency
print("........Standard efficiency Prediction.....")
results_stdd=metrics_test_data(y_tests, MLP_y_preds)

#High efficiency
print("........High efficiency Prediction.....")
results_high=metrics_test_data(y_testh, MLP_y_predh)

#Ultra efficiency
print("........Ultra efficiency Prediction.....")
results_ultra=metrics_test_data(y_testu, MLP_y_predu)

#Non-residential load
print("........Non-residential Load Prediction.....")
results_nonres=metrics_test_data(y_testn, MLP_y_predn)    




#...............................................................................................................................
#RETRAINING ZONE
#Training a new model with the same parameters but using the entire 2016 dataset
#Base case
MLP_modelb2 = MLP(activation= 'relu', hidden_layer_sizes= (1452,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)
#normalize x
normb2 =MinMaxScaler().fit(X)
X_normb2=normb2.transform(X)

#normalize y
ynormb2=MinMaxScaler().fit(y_b)
y_normb2=ynormb2.transform(y_b)

#fit the model
MLP_modelb2.fit(X_normb2, y_normb2)

#..................................................................................................................
# Standard efficiency
MLP_models2 =  MLP(activation= 'relu', hidden_layer_sizes= (1830,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)
#normalize x
norms2 =MinMaxScaler().fit(X)
X_norms2=norms2.transform(X)

#normalize y
ynorms2=MinMaxScaler().fit(y_s)
y_norms2=ynorms2.transform(y_s)

#fit the model
MLP_models2.fit(X_norms2, y_norms2)

#..................................................................................................................
#High efficiency
MLP_modelh2 = MLP(activation= 'relu', hidden_layer_sizes= (1691,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)
#normalize x
normh2 =MinMaxScaler().fit(X)
X_normh2=normh2.transform(X)

#normalize y
ynormh2=MinMaxScaler().fit(y_h)
y_normh2=ynormh2.transform(y_h)

#fit the model
MLP_modelh2.fit(X_normh2, y_normh2)


#..................................................................................................................
# Ultra high efficiency
MLP_modelu2 = MLP(activation= 'relu', hidden_layer_sizes= (1691,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)
#normalize x
normu2 =MinMaxScaler().fit(X)
X_normu2=normu2.transform(X)

#normalize y
ynormu2=MinMaxScaler().fit(y_u)
y_normu2=ynormu2.transform(y_u)

#fit the model
MLP_modelu2.fit(X_normu2, y_normu2)

#..................................................................................................................
# Non-residential Load
MLP_modeln2 = MLP(activation= 'relu', hidden_layer_sizes= (1223,), learning_rate= 'constant', max_iter= 100, solver= 'adam', random_state=42)
#normalize x
normn2 =MinMaxScaler().fit(X_nonres)
X_normn2=normn2.transform(X_nonres)

#normalize y
ynormn2=MinMaxScaler().fit(y_nonres)
y_normn2=ynormn2.transform(y_nonres)

#fit the model
MLP_modeln2.fit(X_normn2, y_normn2)


#.............................................................................................................................................
#TESTING ZONE 

   
df_pred_base=[]
df_pred_high=[]
df_pred_stdd=[]
df_pred_ultra=[]
df_pred_nonres=[]


files=["rcp45cooler_ssp3", "rcp45cooler_ssp5", "rcp45hotter_ssp3", "rcp45hotter_ssp5",
       "rcp85cooler_ssp3", "rcp85cooler_ssp5", "rcp85hotter_ssp3", "rcp85hotter_ssp5"]
for file in files:
    myFiles=glob.glob(file+"/*")
    
    
    
    
    pred_base=[]
    pred_high=[]
    pred_stdd=[]
    pred_ultra=[]
    pred_nonres=[]
    year_name=[]

 
    for data in myFiles:
        
        data_idx = myFiles.index(data)
        
        my_year = int(data[-8:-4])
        year_name.append(my_year)
        
        if my_year !=2099:
            globals()[f'future_data{data[-8:-4]}'] = pd.read_csv(data)
            globals()[f'year_after_data{data[-8:-4]}'] = pd.read_csv(myFiles[data_idx+1])
            df_test_x=pre_process(globals()[f'future_data{data[-8:-4]}'],globals()[f'year_after_data{data[-8:-4]}'],my_year)
            
        else:
            globals()[f'future_data{data[-8:-4]}'] = pd.read_csv(data)
            df_test_x=pre_process(globals()[f'future_data{data[-8:-4]}'],globals()[f'future_data{data[-8:-4]}'],my_year)
         

        
        #normalize test data base case
        df_test_x_normb=normb2.transform(df_test_x.drop(["Weekend"],axis=1))
        y_test_normb=MLP_modelb2.predict(df_test_x_normb).reshape(-1,1)
        #denormalize
        y_test_b = ynormb2.inverse_transform(y_test_normb)
        pred_base.append(y_test_b.flatten())
        
        
        
        #normalize test data standard case
        df_test_x_norms=norms2.transform(df_test_x.drop(["Weekend"],axis=1))
        y_test_norms=MLP_models2.predict(df_test_x_norms).reshape(-1,1)
        #denormalize
        y_test_s=ynorms2.inverse_transform(y_test_norms)
        pred_stdd.append(y_test_s.flatten())
        
        
        #normalize test data high case
        df_test_x_normh=normh2.transform(df_test_x.drop(["Weekend"],axis=1))
        y_test_normh=MLP_modelh2.predict(df_test_x_normh).reshape(-1,1)
        #denormalize
        y_test_h=ynormh2.inverse_transform(y_test_normh)
        pred_high.append(y_test_h.flatten())
        

        #normalize the test data ultra
        df_test_x_normu=normu2.transform(df_test_x.drop(["Weekend"],axis=1))
        y_test_normu=MLP_modelu2.predict(df_test_x_normu).reshape(-1,1)
        #denormalize
        y_test_u=ynormu2.inverse_transform(y_test_normu)
        pred_ultra.append(y_test_u.flatten())
        
        
        #normalize nonres
        df_test_x_normn=normn2.transform(df_test_x)
        y_test_normn=MLP_modeln2.predict(df_test_x_normn).reshape(-1,1)
        #denormalize
        y_test_n=ynormn2.inverse_transform(y_test_normn)
        pred_nonres.append(y_test_n.flatten())
                

        
    result_base=pd.DataFrame(pred_base).transpose()
    result_base.columns=year_name
    result_base= result_base.reindex(sorted(result_base.columns), axis=1)
    result_base.to_csv(str(file)+"_base.csv",index=None) 
    
    result_stdd=pd.DataFrame(pred_stdd).transpose()
    result_stdd.columns=year_name
    result_stdd= result_stdd.reindex(sorted(result_stdd.columns), axis=1)
    result_stdd.to_csv(str(file)+"_stdd.csv",index=None)
    
    result_high=pd.DataFrame(pred_high).transpose()
    result_high.columns=year_name
    result_high= result_high.reindex(sorted(result_high.columns), axis=1)
    result_high.to_csv(str(file)+"_high.csv",index=None)
    
    result_ultra=pd.DataFrame(pred_ultra).transpose()
    result_ultra.columns=year_name
    result_ultra= result_ultra.reindex(sorted(result_ultra.columns), axis=1)
    result_ultra.to_csv(str(file)+"_ultra.csv",index=None)
    
    result_nonres=pd.DataFrame(pred_nonres).transpose()
    result_nonres.columns=year_name
    result_nonres= result_nonres.reindex(sorted(result_nonres.columns), axis=1)
    result_nonres.to_csv(str(file)+"_nonres.csv",index=None)            


# get the end time
et = time.time()
# get the execution time
elapsed_time = (et - st)/3600
print('\nExecution time:', elapsed_time, 'Hours')