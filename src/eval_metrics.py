import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# 评估指标
def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))
    return (tp * (n / p) + tn) / (2 * n)


def test_score_model(preds, y_test, use_zero=False):
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]
    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    preds = preds >= 0
    y_test = y_test >= 0
    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("F1 score: ", f_score)
    print("Accuracy: ", acc)
    print("-" * 50)
    return acc, mae, corr, f_score
