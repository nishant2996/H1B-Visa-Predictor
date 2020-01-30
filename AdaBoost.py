# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from util import func_confusion_matrix
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

def AdaBoostModel(train_X,train_y,val_x,val_y,testX,testY):
    
    # Uncomment this to make the base_estimator as SVM and put base_estimator as svc in AdaBoostClassifier
    #svc = SVC(probability = True, kernel = 'linear')
    
    n_estimatorslist = [25, 50, 100,200]
    accuracy_score = []
    
    for n in n_estimatorslist:
        
        # Running AdaBoost Model using the default base_estimator
        adaClassifier = AdaBoostClassifier(n_estimators = n, learning_rate = 1)
        
        # Training the model using the training dataset
        model = adaClassifier.fit(train_X,train_y)
        
        # Testing the model to predict labels from the test dataset
        y_pred = model.predict(testX)
        
        # Calculating metrics for the model evaluation
        conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(
                                                                testY, y_pred)
        
        print("Confusion Matrix: ")
        print(conf_matrix)
        
        print("Average Accuracy: {}\n".format(accuracy))
        accuracy_score.append(accuracy)
    
        print("Per-Class Precision: {}]\n".format(precision_array))
    
        print("Per-Class Recall: {}".format(recall_array))
        
        if (n == 50):
            
            print("ROC Curve for 50 estimators: \n")
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
            plt.title('ROC Curve for AdaBoost')
            plt.legend(loc="lower right")
            plt.show()
    
    # Graph to compare the accuracy with the number of estimators
    plt.figure()
    plt.plot(n_estimatorslist,accuracy_score)
    plt.ylim([0.0, 1.0])
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.show()
    return accuracy_score[2]
    