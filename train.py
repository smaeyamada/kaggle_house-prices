# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:00:04 2020

original code: Fedi at kaggle
@author: Shin
"""

# Import packages
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
#from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import sklearn.metrics as metrics
import math

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train0 = load_train_data()
    df_test0 = load_test_data()

    logger.info('concat train and test datasets: {} {}'.format(df_train0.shape, df_test0.shape))

    df_train0['train'] = 1
    df_test0['train'] = 0
    df = pd.concat([df_train0, df_test0], axis=0, sort=False)

    logger.info('Data preprocessing')

    # Drop PoolQC, MiscFeature, Alley and Fence features
    # because they have more than 80% of missing values.
    df = df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

    object_columns_df = df.select_dtypes(include=['object'])
    numerical_columns_df =df.select_dtypes(exclude=['object'])

    columns_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
    object_columns_df[columns_None]= object_columns_df[columns_None].fillna('None')

    columns_with_lowNA = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']
    # fill missing values for each column (using its own most frequent value)
    object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])

    diff_year = (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt']).median()
    med_LotFrontage = numerical_columns_df["LotFrontage"].median()
    logger.info(f'Year from built:{diff_year}')
    logger.info(f'median LogFrontage:{med_LotFrontage}')
    numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold']-35)
    numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(68)
    numerical_columns_df= numerical_columns_df.fillna(0)

    object_columns_df = object_columns_df.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)

    # Now we will create some new features
    numerical_columns_df.loc[numerical_columns_df['YrSold'] < numerical_columns_df['YearBuilt'],'YrSold' ] = 2009
    numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])

    numerical_columns_df['TotalBsmtBath'] = numerical_columns_df['BsmtFullBath'] + numerical_columns_df['BsmtFullBath']*0.5
    numerical_columns_df['TotalBath'] = numerical_columns_df['FullBath'] + numerical_columns_df['HalfBath']*0.5
    numerical_columns_df['TotalSA']=numerical_columns_df['TotalBsmtSF'] + numerical_columns_df['1stFlrSF'] + numerical_columns_df['2ndFlrSF']

    # Now the next step is to encode categorical features
    # Ordinal categories features - Mapping from 0 to N
    bin_map  = {'TA':2, 'Fa':1, 'Ex':4, 'Po':1, 'None':0, 'Y':1, 'N':0, 'Reg':3, 'IR1':2, 'IR2':1, 'IR3':0
                "No":2, "Mn":2, "Av":3, "Gd":4, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6
            }
    object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)
    object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)
    object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)
    object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)
    object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)
    object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)
    object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)
    object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)
    object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)
    object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)
    object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)
    object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)
    object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)
    object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)

    PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}
    object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PavedDrive)

    # Will we use One hot encoder to encode the rest of categorical features
    rest_object_columns = object_columns_df.select_dtypes(include=['object'])
    object_columns_df = pd.get_dummies(object_columns_df, columns=rest_object_columns.columns)

    # Concat Categorical(after encoding) and numerical features
    df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1,sort=False)

    df_final = df_final.drop(['Id',],axis=1)
    df_train = df_final[df_final['train'] == 1].drop(['train'], axis=1)
    df_test = df_final[df_final['train'] == 0].drop(['SalePrice','train'], axis=1)

    # Separate Train and Targets
    target= df_train['SalePrice']
    df_train = df_train.drop(['SalePrice'],axis=1)

    logger.info('Modeling')

    x_train,x_test,y_train,y_test = train_test_split(df_train,target,test_size=0.33,random_state=0)

    xgb = XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.02, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2000,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1,
             silent=None, subsample=0.8, verbosity=1)


    lgbm = LGBMRegressor(objective='regression',
                                       num_leaves=4,
                                       learning_rate=0.01,
                                       n_estimators=12000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.4,
                                       )

    # Fitting
    xgb.fit(x_train, y_train)
    lgbm.fit(x_train, y_train,eval_metric='rmse')

    pred_xgb = xgb.predict(x_test)
    pred_lgb = lgbm.predict(x_test)

    logger.info('Root Mean Square Error test(xgb)  = ' + str(math.sqrt(metrics.mean_squared_error(y_test, pred_xgb))))
    logger.info('Root Mean Square Error test(lgbm) = ' + str(math.sqrt(metrics.mean_squared_error(y_test, pred_lgb))))

    xgb.fit(df_train, target)
    lgbm.fit(df_train, target,eval_metric='rmse')

    logger.info('train end')

    pred_lgb = lgbm.predict(df_test)
    pred_xgb = xgb.predict(df_test)
    predict_y = ( pred_xgb * 0.45 + pred_lgb * 0.55)

    df_submit = pd.DataFrame({
            "Id": df_test0["Id"],
            "SalePrice": predict_y
        })
    df_submit.to_csv(DIR + 'submit_200810.csv', index=False)

    logger.info('end')
