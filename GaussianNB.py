# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from util import func_confusion_matrix
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

def GaussianNBModel(train_X,train_y,val_x,val_y,testX,testY):
    
    # Calling the GaussianNB model
    classifier = GaussianNB()
    
    # Training the model on the train dataset
    classifier.fit(train_X,train_y)
    
    # Testing the model on the test dataset
    y_pred = classifier.predict(testX)
    
    #Calculating the metrics for the evaluation of the model
    conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(
                                                                testY, y_pred)
    
    # Plotting the ROC Curve
    fpr, tpr, thresholds = roc_curve(y_pred, testY)
    roc_auc = auc(fpr,tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for GaussianNB')
    plt.legend(loc="lower right")
    plt.show()
    
    return accuracy, precision_array, recall_array, conf_matrix
    