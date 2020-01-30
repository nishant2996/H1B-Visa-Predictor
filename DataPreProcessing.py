# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def case_submission_year_range(df):
    df['CASE_SUBMITTED_YEAR_RANGE'] = np.nan
    for i in range(len(df['CASE_SUBMITTED_YEAR'])):
        if int(df['CASE_SUBMITTED_YEAR'][i]) <= 2012:
            df.loc[i, 'CASE_SUBMITTED_YEAR_RANGE'] = 'BEFORE 2012'
        if int(df['CASE_SUBMITTED_YEAR'][i]) > 2012:
            df.loc[i, 'CASE_SUBMITTED_YEAR_RANGE']  = 'AFTER 2012'
			
def prevailing_wage(df):
    df['PREVAILING_WAGE_RANGE'] = np.nan
    for i in range(len(df['PREVAILING_WAGE'])):
        if int(df['PREVAILING_WAGE'][i]) <= 20000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] = '0 - 20000'
        if int(df['PREVAILING_WAGE'][i]) > 20000 and int(df['PREVAILING_WAGE'][i]) <= 50000:
            df.loc[i, 'PREVAILING_WAGE_RANGE']  = '20000 - 50000'
        if int(df['PREVAILING_WAGE'][i]) > 50000 and int(df['PREVAILING_WAGE'][i]) <= 120000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] = '50000 - 120000'
        if int(df['PREVAILING_WAGE'][i]) > 120000 and int(df['PREVAILING_WAGE'][i]) <= 250000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] = '120000 - 250000'
        if int(df['PREVAILING_WAGE'][i]) > 250000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] ='>250000'

def merge_labels(df):
    
    df['CASE_STATUS'] = df['CASE_STATUS'].replace(['WITHDRAWN'], ['DENIED'])
    df['CASE_STATUS'] = df['CASE_STATUS'].replace(['CERTIFIEDWITHDRAWN'], ['CERTIFIED'])


def fill_nan_values(df):
    '''The dataset consists of many nan values. These are replaced by the mode 
    for various columns like EMPLOYER_NAME,
    EMPLOYER_STATE, FULL_TIME_POSITION ,PW_UNIT_OF_PAY ,PW_SOURCE, PW_SOURCE_YEAR, 
    H-1B_DEPENDENT, WILLFUL_VIOLATOR. For the column PREVAILING_WAGE we replace
    the nan columns with the mean value of the wage data. Also, if the SOC_NAME () 
    is not available, we replace it with hardcoded value Others'''

    df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].fillna(df['EMPLOYER_NAME'].mode()[0])
    df['EMPLOYER_STATE'] = df['EMPLOYER_STATE'].fillna(df['EMPLOYER_STATE'].mode()[0])
    df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].fillna(df['FULL_TIME_POSITION'].mode()[0])
    df['PW_UNIT_OF_PAY'] = df['PW_UNIT_OF_PAY'].fillna(df['PW_UNIT_OF_PAY'].mode()[0])
    df['PW_SOURCE'] = df['PW_SOURCE'].fillna(df['PW_SOURCE'].mode()[0])
    df['PW_SOURCE_YEAR'] = df['PW_SOURCE_YEAR'].fillna(df['PW_SOURCE_YEAR'].mode()[0])
    df['H-1B_DEPENDENT'] = df['H-1B_DEPENDENT'].fillna(df['H-1B_DEPENDENT'].mode()[0])
    df['WILLFUL_VIOLATOR'] = df['WILLFUL_VIOLATOR'].fillna(df['WILLFUL_VIOLATOR'].mode()[0])


    df['SOC_NAME'] = df.SOC_NAME.replace(np.nan, 'Others', regex=True)

    df.PREVAILING_WAGE.fillna(df.PREVAILING_WAGE.mean(), inplace=True)


def classify_employer(df):

    # Broadly classifying the occupations for people filing the visa petition
    df.loc[df['SOC_NAME'].str.contains('COMPUTER OCCUPATION|GRAPHIC DESIGNERS|ANALYSTS'),'SOC_NAME'] = 'IT INDUSTRY'
    df.loc[df['SOC_NAME'].str.contains('ACCOUNTANTS|BUSINESS OPERATIONS SPECIALIST|CHIEF EXECUTIVES|CURATORS|EVENT PLANNERS|FIRST LINE SUPERVISORS|HUMAN RESOURCES|IT MANAGERS|MANAGEMENT|MANAGERS|PUBLIC RELATIONS'),'SOC_NAME'] = 'MANAGEMENT'
    df.loc[df['SOC_NAME'].str.contains('ACTUARIES|FINANCE'),'SOC_NAME'] = 'FINANCE'
    df.loc[df['SOC_NAME'].str.contains('AGRICULTURE|ANIMAL HUSBANDARY|FOOD PREPARATION WORKERS'),'SOC_NAME'] = 'FOOD AND AGRICULTURE'
    df.loc[df['SOC_NAME'].str.contains('COACHES AND SCOUTS|COUNSELORS|EDUCATION|FITNESS TRAINERS|INTERPRETERS AND TRANSLATORS|LIBRARIANS|LOGISTICIANS|SURVEYORS|WRITERS EDITORS AND AUTHORS'),'SOC_NAME'] = 'EDUCATION'
    df.loc[df['SOC_NAME'].str.contains('SALES AND RELATED WORKERS|MARKETING'),'SOC_NAME'] = 'MARKETING'
    df.loc[df['SOC_NAME'].str.contains('DOCTORS|SCIENTIST|INTERNIST'),'SOC_NAME'] = 'ADVANCED SCIENCES'
    df.loc[df['SOC_NAME'].str.contains('COMMUNICATIONS|ENGINEERS|LAB TECHNICIANS|CONSTRUCTION|ARCHITECTURE|MECHANICS'),'SOC_NAME'] = 'ENGINEERING AND ARCHITECTURE'
    df.loc[df['SOC_NAME'].str.contains('DESIGNERS|ENTERTAINMENT|FASHION DESIGNERS|MULTIMEDIA ARTISTS AND ANIMATORS'),'SOC_NAME'] = 'ARTISTS AND ENTERTAINMENT'
    
# Pre-proessing data for the training dataset
def preprocessingTrainingdata():
    
    print("Pre Processing the data for training dataset")
    
    df_train = pd.read_csv('File 1 - H1B Dataset.csv',encoding="ISO-8859-1")

    merge_labels(df_train)

    #clean data by filling the NAN data with appropriate values
    fill_nan_values(df_train)
    prevailing_wage(df_train)
    case_submission_year_range(df_train)
    
    # Creating column UNIV_EMPLOYER for checking if the emplyer name is a university
    classify_employer(df_train)

    class_mapping = {'CERTIFIED': 0, 'DENIED': 1}
    df_train["CASE_STATUS"] = df_train["CASE_STATUS"].map(class_mapping)

    df1_train_set = df_train[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'SOC_NAME', 'WORKSITE_STATE',
     'CASE_STATUS']].copy()

    df1_train_set[['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'SOC_NAME', 'WORKSITE_STATE',
     'CASE_STATUS']] = df1_train_set[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'SOC_NAME', 'WORKSITE_STATE',
     'CASE_STATUS']].apply(lambda x: x.astype('category'))

    X = df1_train_set.loc[:, 'FULL_TIME_POSITION':'WORKSITE_STATE']
    Y = df1_train_set.CASE_STATUS

    seed = 5
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3, random_state=seed)

    X_train_encode = pd.get_dummies(X_train)
    X_val_encode = pd.get_dummies(X_validation)

    train_X = X_train_encode.values
    train_y = Y_train.values

    val_x = X_val_encode.values
    val_y = Y_validation.values
    
    print("Data Pre Processing is completed for training dataset")
    return train_X,train_y,val_x,val_y


# Pre-processing data for the test dataset
def preprocessingTestingdata():
    
    print("Pre Processing data for test dataset")
    
    df_test = pd.read_csv('File 2 - H1B Dataset.csv',encoding="ISO-8859-1")
    
    merge_labels(df_test)
    fill_nan_values(df_test)
    prevailing_wage(df_test)
    case_submission_year_range(df_test)
    classify_employer(df_test)

    df1_test_set = df_test[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 
     'SOC_NAME', 'WORKSITE_STATE','CASE_STATUS']].copy()

    df1_test_set[['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 
     'SOC_NAME', 'WORKSITE_STATE', 'CASE_STATUS']] = df1_test_set[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE',
     'SOC_NAME', 'WORKSITE_STATE', 'CASE_STATUS']].apply(lambda x: x.astype('category'))    
    class_mapping = {'CERTIFIED': 0, 'DENIED': 1}
    
    df1_test_set["CASE_STATUS"] = df1_test_set["CASE_STATUS"].map(class_mapping)

    X_test = df1_test_set.loc[:, 'FULL_TIME_POSITION':'WORKSITE_STATE']
    Y_test = df1_test_set.CASE_STATUS

    X_test_encode = pd.get_dummies(X_test)
    testX = X_test_encode.values
	
    testY = Y_test.values
    print("Data Pre Processing is completed for test dataset")
    return testX, testY