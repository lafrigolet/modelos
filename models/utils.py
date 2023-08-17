import math
import sklearn.metrics as SKL
import matplotlib.pyplot as PLT
import custom_dataset

def roc_curve(results, labels):
    x = [math.exp(result[1]) for result in results]
    y = labels
        
    fpr, tpr, thresholds = SKL.roc_curve(y,x)
    PLT.plot(fpr,tpr,color="blue")
    PLT.grid()
    PLT.xlabel("FPR (especifidad)", fontsize=12, labelpad=10)
    PLT.ylabel("TPR (sensibilidad, Recall)", fontsize=12, labelpad=10)
    PLT.title("ROC de suicidios", fontsize=14)
    
    nlabels = int(len(thresholds) / 5)
 
    for cont in range(0,len(thresholds)):
        if not cont % nlabels:
            PLT.text(fpr[cont], tpr[cont], "  {:.2f}".format(thresholds[cont]),color="blue")
            PLT.plot(fpr[cont], tpr[cont],"o",color="blue")

    PLT.show()

    return x, y


