# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from DataPreProcessing import merge_labels,fill_nan_values, prevailing_wage,case_submission_year_range,classify_employer, preprocessingTrainingdata,preprocessingTestingdata
from Apriori import Apriori
from GaussianNB import GaussianNBModel
from kMeans import kMeansModel
from AdaBoost import AdaBoostModel
import time


# Plotting the graph for the comparison of accuracy of models
def drawResults():
    fig,ax=fig, ax = plt.subplots()
    index=np.arange(len(accuracyList))
    print(index)
    width = 0.35  # the width of the bars
    rects1 = ax.bar(index - width/2, accuracyList, width,
                color='SkyBlue', label='Accuracy')
    ax.set_ylabel('Accuracy Scores')
    ax.set_title('Performance from different models')
    ax.set_xticks(index)
    ax.set_xticklabels(('GaussianNB', 'k-Means','AdaBoost'))
    ax.legend()
    plt.show(rects1)

# Plotting the graph to compare the runtimes of the models
def timeGraph(runtimes):
    models=['GaussianNB','k-Means','AdaBoost']
    plt.plot(models,runtimes,marker='o',color='red')
    plt.title('Time graph for different models')
    plt.ylabel('Time in seconds')
    plt.xlabel('Models')
    plt.xticks(models)
    plt.show()
 
# Visualizing the data to get the relevant attributes for the prediction
def dataVisualization():
    print('Plotting graphs for Data Visualization')
    df = pd.read_csv('File 1 - H1B Dataset.csv',encoding="ISO-8859-1")
    
    # Implementing the pre-processing on the dataset
    classify_employer(df)
    merge_labels(df)
    
    # Converting the labels into binary classification
    certified = df[df['CASE_STATUS']=='CERTIFIED']
    certified = certified['CASE_STATUS'].count()
    denied=df[df['CASE_STATUS']=='DENIED']
    denied=denied['CASE_STATUS'].count()
    
    df1=pd.DataFrame({'Training Data':['Certified','Denied'],'Number of Petitions':[certified,denied]})
    df1.plot.bar(x='Training Data',y='Number of Petitions',rot=0,legend=True)
    plt.title('Case Status vs Number of Petitions')
    plt.show()
 
    #determining the petitions per year
    print('The number of applications per year')
    year_df=df.groupby('CASE_SUBMITTED_YEAR').count().reset_index()
    plt.plot(year_df['CASE_SUBMITTED_YEAR'], year_df['CASE_STATUS'])
    plt.xlabel('Year')
    plt.ylabel('Petition count')
    plt.show()
    
    #displaying the spread of salaries to determine its effetiveness using boxplot
    print('Boxplot of salaries of employees with the status of H-1B visa')
    yearly_wage_df=df[df['WAGE_UNIT_OF_PAY']=='Year']
    sns.boxplot(x='CASE_STATUS',y='PREVAILING_WAGE',data=yearly_wage_df)
    plt.show()
    
    #finding significance of Full time postion label using count plot
    print('H-1B visa status depending on the full time position')
    g=sns.countplot(x='FULL_TIME_POSITION',hue='CASE_STATUS',data=df)
    plt.show()
    
    print('Employees having no Full time jobs and the status of their H-1B visa')
    fulltime_df=df[df['FULL_TIME_POSITION']=='N']
    fulltime_df=fulltime_df.groupby('CASE_STATUS').count().reset_index()
    sns.barplot(x='CASE_STATUS',y='FULL_TIME_POSITION',data=fulltime_df)
    plt.show()
    
    #count of H-1B Visa approval depending on the field of employee using count plot
    print('The H-1B Visa status depending on the field of their job')
    g=sns.countplot(x='SOC_NAME',hue='CASE_STATUS',data=df)
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    plt.legend(loc='upper center')
    plt.show()
    
    #significance of dependent on the H-1B Visa approval using countplot
    print('Count plot of H-1B visa status depending on the dependents')
    g=sns.countplot(x='H-1B_DEPENDENT',hue='CASE_STATUS',data=df)
    plt.show()
    
    #finding whether the worksite state is important using coutnplot
    figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    print('Count plot of H-1B Visa status according to worksite state')
    g=sns.countplot(x='WORKSITE_STATE',hue='CASE_STATUS',data=df)
    plt.legend(loc='upper center')
    plt.show()
    
if __name__ == "__main__":
    accuracyList=[]
    runtimes=[]
    isGaussianNB = False
    iskMeans = False
    isAdaBoost = False
    
    # Visualization of the dataset
    dataVisualization()
    
    # Pre-processing the training and testing data for the models
    train_x,train_y,val_x,val_y=preprocessingTrainingdata()
    testX,testY=preprocessingTestingdata()
    
    while True:
        print("Select the model below:")
        print('''(1) GaussianNB\n(2) kMeans \n(3) AdaBoost \n(4) Show graphs to compare models\n(5) Association Rules\n(6) Exit''')
        
        user_input=input('Enter your choice:')
            
        # GaussianNB model
        if(user_input=='1'):
            isGaussianNB=True
            print('GaussianNB model selected')
            start_time=time.time()
            g_accuracy, precision_array, recall_array, conf_matrix=GaussianNBModel(train_x,train_y,val_x,val_y,testX,testY)
            print("Confusion Matrix: ")
            print(conf_matrix)
    
            print("Accuracy: {} \n".format(g_accuracy))
    
            print("Per-Class Precision: {} \n".format(precision_array))
    
            print("Per-Class Recall: {}".format(recall_array))
            g_runtime = time.time()-start_time
            print("Runtime is {} seconds".format(float(g_runtime)))
            runtimes.append(g_runtime)
            accuracyList.append(g_accuracy)
        
        # k-Means model
        elif(user_input=='2'):
            iskMeans=True
            print('kMeans model selected')
            #train_x,train_y,val_x,val_y=preprocessingTrainingdata(user_input)
            #testX,testY=preprocessingTestingdata(user_input)
            start_time=time.time()
            k_accuracy=kMeansModel(train_x,train_y,val_x,val_y,testX,testY)
            print("Highest accuracy:{} \n".format(k_accuracy))
            k_runtime = time.time()-start_time
            print("Runtime is {} seconds".format(float(g_runtime)))
            runtimes.append(k_runtime)
            accuracyList.append(k_accuracy)
            
        # Adaboost model
        elif(user_input=='3'):
            isAdaBoost=True
            print('AdaBoost model selected')
            #train_x,train_y,val_x,val_y = preprocessingTrainingdata(user_input)
            #testX,testY = preprocessingTestingdata(user_input)
            start_time=time.time()
            a_accuracy=AdaBoostModel(train_x,train_y,val_x,val_y,testX,testY)
            print("Highest accuracy:{} \n".format(a_accuracy))
            a_runtime = time.time()-start_time
            print("Runtime is {} seconds".format(float(a_runtime)))
            runtimes.append(a_runtime)
            accuracyList.append(a_accuracy)
            
        # Generating the graphs for the comparison of models
        elif(user_input=='4'):
            if(len(runtimes)==3):
                timeGraph(runtimes)
                drawResults()
            else:
                print('Please run all the models first')
                break
        
        elif(user_input=='5'):
            Apriori()
        
        elif(user_input=='6'):
            print("Thanks for using the program good byee!!")
            break
        else:    
            print("Not a valid input, program is exiting bye!!")
            break