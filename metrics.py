# Evaluation
from sklearn.metrics import roc_auc_score
import torch
from torchmetrics import F1Score
from torchmetrics import ROC
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional import auc
from torchmetrics.functional import precision_recall
from torchmetrics.functional import auc

## Binary Classification
def bin_class_metrics(pred, target, positive_class = 1):
    conf_mat = ConfusionMatrix(num_classes=2)
    cm = conf_mat(pred, target)
    print("Confusion Matrix (0 in Top Left): ")
    print(cm)
    
    if positive_class == 1: 
        # true positives / (true positives + false positives)
        # recall = true positives / (true positives + false negatives)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
    else:
        tp = cm[0][0]
        fp = cm[1][0]
        fn = cm[0][1]
      
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * precision * recall / (precision + recall)
    
    if not ((precision.item() > 0) & (precision.item() < 100)):
        precision = 0
        print("\n\nprecision is nan (has been set to 0)")
    
    if not ((recall.item() > 0) & (recall.item() < 100)):
        recall = 0
        print("recall is nan (has been set to 0)")
    
    if not ((f1score.item() > 0) & (f1score.item() < 100)):
        f1score = 0
        print("F1score is nan (has been set to 0)\n\n")
    
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    
#     roc = ROC(num_classes = 2, pos_label=1)
#     score_fpr, score_tpr, _ = roc(target, pred)
#     score_roc_auc = roc_auc_score(target, pred)
    
## Multiclass Classification
def multiclass_metrics(pred, target, num_classes):
    conf_mat = ConfusionMatrix(num_classes=num_classes)
    cm = conf_mat(pred, target)
    print("Confusion Matrix (0 in Top Left): ")
    print(cm)


    metric = MulticlassF1Score(num_classes=num_classes, average='macro')
    f1_score_avg = metric(pred, target)
    print("F1-Score (Average)", f1_score_avg)

    metric = MulticlassF1Score(num_classes=num_classes, average=None)
    f1_score_each = metric(pred, target)
    print("F1-Score (each):")
    for i, f in enumerate(f1_score_each):
        print("Class ", i, ":", f)
        
def evaluate_model_metrics(model, num_class, dataloader):
    # to store all labels and predictions for f1-score
    all_pred = []
    all_label = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            # calculate outputs by running images through the network
            outputs, h = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            all_pred.append(predicted)
            all_label.append(labels)

    # get all predictions and labels into one array and as integer tensors
    all_pred = [i for s in all_pred for i in s]
    all_label = [i for s in all_label for i in s]
    all_label = [all_label[i][0] for i in range(0, len(all_label))]
    all_label = [all_label[i].to(dtype=torch.long) for i in range(len(all_label))]

    all_pred = torch.LongTensor(all_pred)
    all_label = torch.LongTensor(all_label)
    if num_class == 2:
        bin_class_metrics(all_pred, all_label, positive_class = 1)
    elif num_class > 2:
        multiclass_metrics(all_pred, all_label, num_class)
    return all_pred

def print_metrics(model, model_name, num_classes, train_dl, test_dl):
    print("-----------------------------------------------------------------------------------")
    print(model_name.upper(), " Metrics")
    print("Train")
    preds_train = evaluate_model_metrics(model, num_classes, train_dl)

    print("Test")
    preds_test = evaluate_model_metrics(model, num_classes, test_dl)

    print("-----------------------------------------------------------------------------------")
    
def PlotLoss(train_loss, val_loss, model_name):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1,len(val_loss)+1), val_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = val_loss.index(min(val_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.title("{} Loss for Hidden Size: {} and Batch Size: {}".format(model_name.upper(), h, b))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.2) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('./loss_plots/{}_loss_plot_{}.png'.format(model_name, output_size), bbox_inches='tight')