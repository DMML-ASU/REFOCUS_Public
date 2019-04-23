from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, roc_auc_score

def refocus(in_file, beta, clf):
    print "One Classifier on the whole training set and max F{} as Criteria".format(beta)
    print "Dataset: {}".format(in_file)
    print clf
    X, y = make_X_y(in_file)
    sss = StratifiedShuffleSplit(random_state=0, n_splits=10)

    gl_t = []
    gl_p = []
    gl_r = []
    gl_f1 = []
    gl_roc = []
    gl_cm = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t = []
        for train_index_1, test_index_1 in sss.split(X_train, y_train):
            X_train_1, X_val = X_train[train_index_1], X_train[test_index_1]
            y_train_1, y_val = y_train[train_index_1], y_train[test_index_1]
            clf.fit(X_train_1, y_train_1)
            y_pred_prob = clf.predict_proba(X_val)
            f = []
            for threshold in np.arange(0.1, 1, 0.1):
                y_pred = []
                for elem in y_pred_prob:
                    if elem[1] > threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(-1)
                result = precision_recall_fscore_support(y_val, y_pred, average='binary', pos_label=1, beta=beta)
                f.append(result[2])
            threshold = 0.1 + (np.argmax(f))*0.1
            t.append(threshold)

        clf.fit(X_train, y_train)
        y_pred_proba_test = clf.predict_proba(X_test)
        y_pred_test = []
        y_pred_test_for_roc = []
        threshold = np.mean(t)
        gl_t.append(threshold)
        for elem in y_pred_proba_test:
            if elem[1] > threshold:
                y_pred_test.append(1)
            else:
                y_pred_test.append(-1)
        for elem in y_pred_proba_test:
            y_pred_test_for_roc.append(elem[1])

        result = precision_recall_fscore_support(y_test, y_pred_test, average='binary', pos_label=1, beta=1)
        print "On Test Set - Threshold {}: p={}, r={}, f1={}, roc={}".format(threshold, result[0], result[1], result[2], roc_auc_score(y_test, y_pred_test_for_roc))
        gl_p.append(result[0])
        gl_r.append(result[1])
        gl_f1.append(result[2])
        c = confusion_matrix(y_test, y_pred_test)
        c = c.astype('float') / c.sum(axis=1)[:, np.newaxis]
        gl_cm.append(c)
        gl_roc.append(roc_auc_score(y_test, y_pred_test_for_roc))
    print "p:{}, r:{}, f1:{}, roc:{}, threshold={}".format(np.mean(gl_p), np.mean(gl_r), np.mean(gl_f1), np.mean(gl_roc), np.mean(gl_t))
    print np.mean(gl_cm, axis=0)

def baseline(in_file):
    print "Running SVM on {}".format(in_file)
    X, y = make_X_y(in_file)

    sss = StratifiedShuffleSplit(random_state=0, n_splits=10)
    clf = SVC(random_state=0, probability=True, gamma='auto')

    gl_t = []
    gl_p = []
    gl_r = []
    gl_f1 = []
    gl_roc = []
    gl_cm = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred_proba_test = clf.predict_proba(X_test)
        y_pred_test = []
        y_pred_test_for_roc = []
        threshold = 0.5
        gl_t.append(threshold)
        for elem in y_pred_proba_test:
            if elem[1] > threshold:
                y_pred_test.append(1)
            else:
                y_pred_test.append(-1)
        for elem in y_pred_proba_test:
            y_pred_test_for_roc.append(elem[1])

        result = precision_recall_fscore_support(y_test, y_pred_test, average='binary', pos_label=1, beta=1)
        print "On Test Set - Threshold {}: p={}, r={}, f1={}, roc={}".format(threshold, result[0], result[1], result[2], roc_auc_score(y_test, y_pred_test_for_roc))
        gl_p.append(result[0])
        gl_r.append(result[1])
        gl_f1.append(result[2])
        c = confusion_matrix(y_test, y_pred_test)
        c = c.astype('float') / c.sum(axis=1)[:, np.newaxis]
        gl_cm.append(c)
        gl_roc.append(roc_auc_score(y_test, y_pred_test_for_roc))
    print "p:{}, r:{}, f1:{}, roc:{}, threshold={}".format(np.mean(gl_p), np.mean(gl_r), np.mean(gl_f1), np.mean(gl_roc), np.mean(gl_t))
    print np.mean(gl_cm, axis=0)

if __name__ == '__main__':
    clf_svm = SVC(random_state=0, probability=True, gamma='auto')
    clf_dt  = DecisionTreeClassifier(random_state=0)
    clf_rf  = RandomForestClassifier(random_state=0, max_depth=1)
    clf_lr  = LogisticRegression(random_state=0, max_iter=1)
    refocus(in_file='<an LDA probability distribution file>', beta=2, clf=clf_svm)
    baseline(in_file='<an LDA probability distribution file>')