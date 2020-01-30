# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from util import func_confusion_matrix
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

def kMeansModel(train_X,train_y,val_x,val_y,testX,testY):
    
    n_clusters = [2,3,4,5]
    accuracy_score = []
    
    for n in n_clusters:
        
        # Calling the kMeans function from the library
        kmeans = KMeans(n)
        
        # Training the model to form the clusters
        kmeans = kmeans.fit(train_X)
        
        # Predicting the labels for the test dataset
        y_pred = kmeans.predict(testX)
        
        # Calculationg metrics for the model evaluation
        conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(
                                                                testY, y_pred)
        
        print("Evaluation metrics for {} clusters".format(n))
        print("Confusion Matrix: ")
        print(conf_matrix)
        
        print("Average Accuracy: {}\n".format(accuracy))
        accuracy_score.append(accuracy)
    
        print("Per-Class Precision: {}]\n".format(precision_array))
    
        print("Per-Class Recall: {}".format(recall_array))
    
        if (n == 2):
            
            print("For 2 clusters, the ROC Curve is: \n")
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
            plt.title('ROC Curve for k-Means')
            plt.legend(loc="lower right")
            plt.show()
    
    # Graph for the accuracy compared to the number of clusters
    plt.figure()
    plt.plot(n_clusters,accuracy_score)
    plt.ylim([0.0, 1.0])
    plt.xlabel('Number of clusters')
    plt.ylabel('Accuracy')
    plt.show()
    
    return accuracy_score[0]