import os
from os import listdir
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataloader import load_data
from helpers import get_cat_dims
import pandas as pd
from models import WGANGP
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import classification_report
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)

results = {}
epochs = [100]

def append_record(record):
    with open('./results/individual_results_wgangp.txt', 'w') as f:
        f.write(str(record))
        f.write(os.linesep)

path = "./final_datasets/"
for epoch in epochs :
      for name in os.listdir(path) :
        if name.endswith(".csv") :
          for i in range(10) :
            print()
            print()
            print(name, i+1, epoch)
            print()
            print()
            df = pd.read_csv(path+name, header=None)
            df.loc[df[df.columns[-1]]>=1, df.columns[-1]] = 1
            cat_cols = None
            num_cols = list(df.columns[:-1])
            target_col = df.columns[-1]
            if cat_cols is not None :
                X = df.loc[:, num_cols + cat_cols]
            else :
                X = df.loc[:, num_cols]
            y = df.loc[:, target_col]
            if i == 0:
              results[name] = {}
            results[name][i+1] = {"wgangp" : {}}

            while(1) :
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42*i, stratify=y)
                if len(Counter(y_test)) > 1 :
                    break

            num_prep = make_pipeline(SimpleImputer(strategy='mean'),
                                     MinMaxScaler())
            cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                     OneHotEncoder(handle_unknown='ignore', sparse=False))
            prep = ColumnTransformer([
                ('num', num_prep, num_cols)],
                remainder='drop')

            cat_dims = get_cat_dims(X_train, cat_cols)

            X_train_trans = prep.fit_transform(X_train)

            gan = WGANGP(write_to_disk=True, # whether to create an output folder. Plotting will be surpressed if flase
                        compute_metrics_every=1250, print_every=2500, plot_every=10000,
                        num_cols = num_cols, cat_dims=cat_dims,
                        # pass the one hot encoder to the GAN to enable count plots of categorical variables
                        transformer=None,
                        # pass column names to enable
                        cat_cols=cat_cols,
                        use_aux_classifier_loss=True,
                        d_updates_per_g=3, gp_weight=15)

            gan.fit(X_train_trans, y=y_train.values, 
                    condition=True,
                    epochs=epoch,  
                    batch_size=64,
                    netG_kwargs = {'hidden_layer_sizes': (128,64), 
                                    'n_cross_layers': 1,
                                    'cat_activation': 'gumbel_softmax',
                                    'num_activation': 'none',
                                    'condition_num_on_cat': False, 
                                    'noise_dim': 30, 
                                    'normal_noise': False,
                                    'activation':  'leaky_relu',
                                    'reduce_cat_dim': True,
                                    'use_num_hidden_layer': True,
                                    'layer_norm':False,},
                    netD_kwargs = {'hidden_layer_sizes': (128,64,32),
                                    'n_cross_layers': 2,
                                    'embedding_dims': None,
                                    'activation':  'leaky_relu',
                                    'sigmoid_activation': False,
                                    'noisy_num_cols': True,
                                    'layer_norm':True,}
                   )

            X_res, y_res = gan.resample(X_train_trans, y=y_train)
            y_res = np.resize(y_res, (y_res.shape[0], 1))
            result = np.concatenate([X_res, y_res], axis=1)
            df = pd.DataFrame(result)
            df.to_csv(path + "hybrid/hybrid_"+str(i+1)+"_"+name, index=False, header=None)

            X_test_trans = prep.transform(X_test)
            test_y = np.resize(y_test, (y_test.shape[0], 1))

            test = np.concatenate([X_test_trans, test_y], axis=1)
            test = pd.DataFrame(test)
            test.to_csv(path + "test/test_"+str(i+1)+"_"+name, header=None, index=False)

            train_y = np.resize(y_train, (y_train.shape[0], 1))

            train = np.concatenate([X_train_trans, train_y], axis=1)
            train = pd.DataFrame(train)
            train.to_csv(path + "train/train_"+str(i+1)+"_"+name, header=None, index=False)
            clf = KNeighborsClassifier(5)
            clf2 = KNeighborsClassifier(5)
            clf.fit(X_res, y_res)
            preds_oversampled = clf.predict_proba(X_test_trans)[:,1]

            clf2.fit(X_train_trans, y_train)
            preds_imbalanced = clf2.predict_proba(X_test_trans)[:,1]
            hybrid1 = pd.read_csv(path + "hybrid/hybrid_"+str(i+1)+"_"+name, header=None)
            test = pd.read_csv(path + "test/test_"+str(i+1)+"_"+name, header=None)

            test_y = test[test.columns[-1]]
            test_x = test.drop([test.columns[-1]], axis=1)

            wgangp_y = hybrid1[hybrid1.columns[-1]]
            wgangp_x = hybrid1.drop([hybrid1.columns[-1]], axis=1)


            clf = KNeighborsClassifier(5)
            clf.fit(wgangp_x, wgangp_y)
            pred_y = clf.predict(test_x)
            report = classification_report(test_y, pred_y, output_dict=True)
            disp = plot_confusion_matrix(clf, test_x, test_y,
                                          display_labels=[0, 1],
                                          cmap=plt.cm.Blues)
            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            score = report['accuracy']
            f1 = report['macro avg']['f1-score']
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])

            tp = disp.confusion_matrix[1][1]
            fp = disp.confusion_matrix[0][1]
            fn = disp.confusion_matrix[1][0]
            tn = disp.confusion_matrix[0][0]
            results[name][i+1]["wgangp"]['knn'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]

            clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=len(wgangp_x.columns))
            clf.fit(wgangp_x, wgangp_y)
            pred_y = clf.predict(test_x)
            report = classification_report(test_y, pred_y, output_dict=True)
            disp = plot_confusion_matrix(clf, test_x, test_y,
                                          display_labels=[0, 1],
                                          cmap=plt.cm.Blues)
            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            score = report['accuracy']
            f1 = report['macro avg']['f1-score']
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])

            tp = disp.confusion_matrix[1][1]
            fp = disp.confusion_matrix[0][1]
            fn = disp.confusion_matrix[1][0]
            tn = disp.confusion_matrix[0][0]
            results[name][i+1]["wgangp"]['rf'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]


            clf = DecisionTreeClassifier(max_depth=5)
            clf.fit(wgangp_x, wgangp_y)
            pred_y = clf.predict(test_x)
            report = classification_report(test_y, pred_y, output_dict=True)
            disp = plot_confusion_matrix(clf, test_x, test_y,
                                          display_labels=[0, 1],
                                          cmap=plt.cm.Blues)
            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            score = report['accuracy']
            f1 = report['macro avg']['f1-score']
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])

            tp = disp.confusion_matrix[1][1]
            fp = disp.confusion_matrix[0][1]
            fn = disp.confusion_matrix[1][0]
            tn = disp.confusion_matrix[0][0]
            results[name][i+1]["wgangp"]['dt'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]


            clf = GaussianNB()
            clf.fit(wgangp_x, wgangp_y)
            pred_y = clf.predict(test_x)
            report = classification_report(test_y, pred_y, output_dict=True)
            disp = plot_confusion_matrix(clf, test_x, test_y,
                                          display_labels=[0, 1],
                                          cmap=plt.cm.Blues)
            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            score = report['accuracy']
            f1 = report['macro avg']['f1-score']
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])

            tp = disp.confusion_matrix[1][1]
            fp = disp.confusion_matrix[0][1]
            fn = disp.confusion_matrix[1][0]
            tn = disp.confusion_matrix[0][0]
            results[name][i+1]["wgangp"]['nb'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]


            clf = LogisticRegression()
            clf.fit(wgangp_x, wgangp_y)
            pred_y = clf.predict(test_x)
            report = classification_report(test_y, pred_y, output_dict=True)
            disp = plot_confusion_matrix(clf, test_x, test_y,
                                          display_labels=[0, 1],
                                          cmap=plt.cm.Blues)
            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            score = report['accuracy']
            f1 = report['macro avg']['f1-score']
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])

            tp = disp.confusion_matrix[1][1]
            fp = disp.confusion_matrix[0][1]
            fn = disp.confusion_matrix[1][0]
            tn = disp.confusion_matrix[0][0]
            results[name][i+1]["wgangp"]['lr'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]
            append_record(results)

dict1 = results
results = {}

nan = 0
for nam in dict1 :
      if(len(dict1[nam]) < 10) :
          print("there are not 10 iterations for " + nam + ", hence, results will not be calculated for it")
          continue
      results[nam] = {"wgangp":{}}
      for itr in dict1[nam] :
            for typ in dict1[nam][itr] :
                  for classifier in dict1[nam][itr][typ] :
                        if typ not in results[nam] :
                              results[nam][typ] = {}
                        if classifier not in results[nam][typ] :
                              results[nam][typ][classifier] = [[], [], [], [], [], [], [], [], [], [], [], []]

                        i = 0
                        for value in dict1[nam][itr][typ][classifier] :
                              results[nam][typ][classifier][i].append(value)
                              i += 1

final_results = {}
for name in results :
      final_results[name] = {"wgangp":{}}
      for typ in results[name] :
            for classifier in results[name][typ] :
                  final_results[name][typ][classifier] = []
                  for values in results[name][typ][classifier] :
                        final_results[name][typ][classifier].append(sum(values)/len(values))

with open('./results/averaged_results_wgangp.txt', 'w') as f:
    f.write(str(final_results))

print("Results are printed in below order.")
print("accuracy, precision, recall, f1-score, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1")
dict1 = final_results
for name in sorted(dict1) :
      print(name)
      for typ in dict1[name] :
            for classifier in dict1[name][typ] :
                  print(classifier)
                  for value in dict1[name][typ][classifier] :
                        print(round(value, 4), end = " ")
                  print()
            print()
      print()