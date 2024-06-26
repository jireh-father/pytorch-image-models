""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def multiclass_f1_score(pred, target, num_classes=0):
    """
    Compute the F1 score for a multi-class classification problem
    """
    return f1_score(target, pred, average='macro', labels=list(range(num_classes)))


def multiclass_recall_score(pred, target, num_classes=0):
    """
    Compute the recall score for a multi-class classification problem
    """
    return recall_score(target, pred, average='macro', labels=list(range(num_classes)))


def multiclass_precision_score(pred, target, num_classes=0):
    """
    Compute the precision score for a multi-class classification problem
    """
    return precision_score(target, pred, average='macro', labels=list(range(num_classes)))


def get_confusion_matrix(pred, target):
    """
    Compute the confusion matrix for a multi-class classification problem
    """
    return confusion_matrix(target, pred)


def get_classification_report(pred, target, class_names=None):
    """
    Compute the classification report for a multi-class classification problem
    """
    return classification_report(target, pred, target_names=class_names)
