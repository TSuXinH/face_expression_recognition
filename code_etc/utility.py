import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, \
    cohen_kappa_score, hamming_loss

num_class = 7
label_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}


# Part 1: used for visualizing the dataset
def show_image(num_list, data_set, label_set):
    for idx, item in enumerate(num_list):
        plt.subplot(100 + len(num_list) * 10 + idx + 1)
        plt.imshow(data_set[item], cmap='gray')
        plt.axis('off')
        plt.title(label_dict[label_set[item]])
    plt.show()


def randomly_show_image(num, data_set, label_set):
    num_list = random.sample([i for i in range(len(label_set))], num)
    show_image(num_list, data_set, label_set)


# Part 2: used for visualizing the result
def draw_accuracy_loss_curve(accuracy_list, loss_list, saving_path=''):
    x = [i for i in range(len(accuracy_list))]
    plt.subplot(121)
    plt.plot(x, accuracy_list, c='g', label='accuracy curve')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(122)
    plt.plot(x, loss_list, c='r', label='loss curve')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    if saving_path != '':
        plt.savefig(saving_path)
    plt.show()


def draw_confusion_matrix(prediction, ground_truth, in_probability=False, saving_path=''):
    """
    This function is used to show the confusion matrix,
    It will show the probabilities (each entry divided by the sum of its row,
    which represents the number of the truth) if in_probability is True else directly show the numbers.
    If saving, the image won't be shown.
    """
    confusion_mat = np.array(confusion_matrix(ground_truth, prediction))
    confusion_mat = confusion_mat / np.sum(confusion_mat, axis=1) if in_probability else confusion_mat
    plt.imshow(confusion_mat, interpolation='nearest')
    tick_str = list(label_dict[s] for s in range(num_class))
    tick_num = list(range(num_class))
    plt.xticks(tick_num, tick_str, rotation=90)
    plt.yticks(tick_num, tick_str)
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        c = 'black' if i == j else 'w'
        if in_probability:
            plt.text(j, i, '{:.2f}'.format(confusion_mat[i][j]), horizontalalignment='center', color=c)
        else:
            plt.text(j, i, confusion_mat[i][j], horizontalalignment='center', color=c)
    if saving_path != '':
        plt.savefig(saving_path)
        plt.close()
    else:
        plt.show()


def cal_acc(pre, gt):
    return np.equal(pre, gt).sum() / len(pre)


# Calculate the indicators
def cal_params(pre, gt):
    TP, TN, FP, FN = 0, 0, 0, 0
    for idx, item in enumerate(pre):
        item = round(item)
        if item == gt[idx]:
            if gt[idx] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if gt[idx] == 1:
                FN += 1
            else:
                FP += 1
    return TP, TN, FP, FN


def calculate_evaluation(pre, gt):
    [TP, TN, FP, FN] = cal_params(pre, gt)
    auROC = roc_auc_score(gt, pre, multi_class='ovo')
    auPRC = average_precision_score(gt, pre)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return {
        'BER': .5 * (FP / (FP + TN) + FN / (FN + TP)),
        'MCC': (TP * TN - FP * FN) / np.sqrt((TP + FP) * (FP + TN) * (TN + FN) * (FN + TP)),
        'sensitivity': TP / (TP + FN),
        'specificity': TN / (TN + FP),
        'recall': recall,
        'precision': precision,
        'F1': 2 * recall * precision / (recall + precision),
        'auROC': auROC,
        'auPRC': auPRC,
    }


def cal_single_eval(pre, gt, k: int):
    """
    k represents choosing the k 'th class for calculating, range: [0, 7)
    """
    assert 0 <= k < num_class
    alter_pre, alter_gt = np.zeros_like(pre), np.zeros_like(gt)
    alter_pre[pre == k] = 1
    alter_gt[gt == k] = 1
    return calculate_evaluation(alter_pre, alter_gt)


def cal_single_label_eval(pre, gt):
    eval_dict_list = []
    for i in range(num_class):
        eval_dict_list.append(cal_single_eval(pre, gt, i))
    output_dict = {
        'BER': .0,
        'MCC': .0,
        'sensitivity': .0,
        'specificity': .0,
        'recall': .0,
        'precision': .0,
        'F1': .0,
        'auROC': .0,
        'auPRC': .0,
    }
    for item in eval_dict_list:
        for item_ in output_dict.keys():
            output_dict[item_] += item[item_]
    for item in output_dict.keys():
        output_dict[item] /= num_class
    return output_dict


def cal_multi_label_eval(pre, gt):
    output_eval = {
        'kappa': cohen_kappa_score(gt, pre),
        'hamming': hamming_loss(gt, pre),
    }
    return output_eval


def cal_all_eval(pre, gt):
    eval_1 = cal_single_label_eval(pre, gt)
    eval_2 = cal_multi_label_eval(pre, gt)
    return {**eval_1, **eval_2}


def print_eval(evaluation):
    print('********************')
    print('evaluating indicators are as below:')
    for key, value in evaluation.items():
        print('{}: {:.6f}'.format(key, value))
    print('********************')


# Others
def get_alpha(label):
    """
    Get the important parameter alpha for adaptive loss.
    This method is seldom used as directly modifying alpha is easier.
    """
    output = []
    for i in range(num_class):
        output.append(1 - len(np.where(label == i)[0]) / len(label))
    return np.array(output)
