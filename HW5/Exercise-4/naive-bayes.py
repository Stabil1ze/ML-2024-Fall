import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def load_data(path):
    """ 加载数据, 计算先验概率, 返回单词表与整合的文本"""

    v = []
    msg = 0  # 非垃圾邮件计数
    sp_msg = 0  # 垃圾邮件计数
    text_nsp = []
    text_sp = []

    for roots, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(roots, file), 'r') as f:
                text = f.read()
                pure_text = re.findall(r'[a-zA-Z]+', text)
                v.extend(pure_text)
                # 通过文件名判断是否为垃圾邮件, 更新单词表和计数
                if 'sp' in file:
                    text_sp.extend(pure_text)
                    sp_msg += 1
                else:
                    text_nsp.extend(pure_text)
                    msg += 1

    # 计算垃圾邮件出现的先验概率
    prior_prob_sp = sp_msg / (msg + sp_msg)
    v = list(set(v))  # 去重
    return v, prior_prob_sp, text_nsp, text_sp


def train_naive_bayes(v, text_nsp, text_sp):
    """ 训练朴素贝叶斯模型, 返回词频和条件概率 """

    n_sp = len(text_sp)
    n_nsp = len(text_nsp)
    len_v = len(v)
    i = 0  # 显示进度

    # 计算词频和条件概率
    word_count_sp = {}
    word_count_nsp = {}
    p_word_sp = {}
    p_word_nsp = {}
    for word in v:
        i += 1
        print('\rTraining Naive Bayes: {:.2f}%'.format(i / len_v * 100), end='')
        word_count_sp[word] = text_sp.count(word)  # n_sp,k
        word_count_nsp[word] = text_nsp.count(word)  # n_nsp,k
        p_word_sp[word] = (word_count_sp[word] + 1) / (n_sp + len_v)  # P(w_k|sp)
        p_word_nsp[word] = (word_count_nsp[word] + 1) / (n_nsp + len_v)  # P(w_k|nsp)

    print('\n')
    return p_word_sp, p_word_nsp


def test_naive_bayes(email, v, prior_prob_sp, p_word_sp, p_word_nsp):
    """ 测试朴素贝叶斯模型, 返回预测结果 """

    # 将乘积转化为对数形式, 避免下溢出
    p_sp = np.log(prior_prob_sp)
    p_nsp = np.log(1 - prior_prob_sp)

    # 计算邮件的后验概率
    pure_text = re.findall(r'[a-zA-Z]+', email)
    for word in pure_text:
        if word in v:
            p_sp += np.log(p_word_sp[word])
            p_nsp += np.log(p_word_nsp[word])
    if p_sp > p_nsp:
        return 1
    else:
        return 0


def test_obj(path):
    """ 读取测试集标签 """
    labels = []
    for roots, dirs, files in os.walk(path):
        for file in files:
            if 'sp' in file:
                labels.append(1)
            else:
                labels.append(0)
    return labels


V, Prior_prob_sp, Text_nsp, Text_sp = load_data('train-mails')
P_word_sp, P_word_nsp = train_naive_bayes(V, Text_nsp, Text_sp)
Labels = test_obj('test-mails')
Predictions = []
Len_v = len(os.listdir('test-mails'))
progress = 0  # 显示进度

# 测试朴素贝叶斯模型
for Roots, Dirs, Files in os.walk('test-mails'):
    for File in Files:
        with open(os.path.join(Roots, File), 'r') as F:
            progress += 1
            print('\rTesting Naive Bayes: {:.2f}%'.format(progress / Len_v * 100), end='')
            Email = F.read()
            Predictions.append(test_naive_bayes(Email, V, Prior_prob_sp, P_word_sp, P_word_nsp))

# 显示结果
print('Confusion Matrix:\n', confusion_matrix(Labels, Predictions))
print('Accuracy:', accuracy_score(Labels, Predictions))
print('Precision:', precision_score(Labels, Predictions))
print('Recall:', recall_score(Labels, Predictions))
print('F1-score:', f1_score(Labels, Predictions))
for n in range(len(Labels)):
    if Labels[n] != Predictions[n]:
        print("Found a wrong prediction in position:" + str(n))
