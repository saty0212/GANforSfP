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
from sklearn.metrics import classification_report
import json
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from vanilla_gan import GAN
import warnings
warnings.filterwarnings('ignore')

results = {}

def append_record(record):
    with open('./results/individual_results_vanilla_gan.txt', 'w') as f:
        f.write(str(record))
        f.write(os.linesep)

path = "./final_datasets/"
for name in os.listdir(path) :
  if name.endswith(".csv") :
    for i in range(10) :
      print()
      print()
      print(name, i+1)
      print()
      print()
      file = pd.read_csv(path+name, header=None)
      file.loc[file[file.columns[-1]]>=1, file.columns[-1]] = 1
      scaler = MinMaxScaler()
      scaler.fit(file)
      file = scaler.transform(file)
      file = pd.DataFrame(file)
      if i == 0:
        results[name] = {}
      results[name][i+1] = {"vanilla_gan" : {}}

      y = file[file.columns[-1]]
      x = file.drop([file.columns[-1]], axis=1)
      
      train_x, test_x, train_y, test_y = tts(x, y, test_size=0.3, shuffle=True, stratify=y)
      train_y = np.resize(train_y, (train_y.shape[0], 1))
      train = np.concatenate([train_x, train_y], axis=1)
      train = pd.DataFrame(train)
      test_y = np.resize(test_y, (test_y.shape[0], 1))
      test = np.concatenate([test_x, test_y], axis=1)
      test = pd.DataFrame(test)
      train_path = path + "train/train_"+str(i+1)+"_"+name
      test_path = path + "test/test_"+str(i+1)+"_"+name
      train.to_csv(train_path, index=False, header=None)
      test.to_csv(test_path, index=False, header=None)

      train = pd.read_csv(train_path, header=None)
      test = pd.read_csv(test_path, header=None)

      test_y = test[test.columns[-1]]
      test_x = test.drop([test.columns[-1]], axis=1)

      nums = Counter(train[train.columns[-1]])
      if nums[0] > nums[1] :
        minor = 1
      else :
        minor = 0

      minority = train[train[train.columns[-1]] == minor]
      minority = minority.drop([minority.columns[-1]], axis=1)

      data_cols = minority.columns
      #Define the GAN and training parameters
      noise_dim = 32
      dim = 128
      batch_size = 32

      log_step = 100
      epochs = 5000+1
      learning_rate = 5e-4
      models_dir = 'model'
      gan_args = [batch_size, learning_rate, noise_dim, minority.shape[1], dim]
      train_args = ['', epochs, log_step]

      model = GAN
      synthesizer = model(gan_args)
      synthesizer.train(minority, train_args)

      z = np.random.normal(size=(abs(nums[0]-nums[1]), noise_dim))
      g_z = synthesizer.generator.predict(z)
      gen_samples = pd.DataFrame(g_z)
      gen_samples[len(train.columns)-1] = minor

      hybrid = pd.concat([gen_samples, train], axis=0, ignore_index=True)
      hybrid.to_csv(path + "hybrid/hybrid_vanilla_gan_"+str(i+1)+"_"+name, index=False, header=None)

      vanilla_gan_y = hybrid[hybrid.columns[-1]]
      vanilla_gan_x = hybrid.drop([hybrid.columns[-1]], axis=1)

      # CLASSIFICATION
      clf = KNeighborsClassifier(5)
      clf.fit(vanilla_gan_x, vanilla_gan_y)
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
      results[name][i+1]["vanilla_gan"]['knn'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]

      clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=len(vanilla_gan_x.columns))
      clf.fit(vanilla_gan_x, vanilla_gan_y)
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
      results[name][i+1]["vanilla_gan"]['rf'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]


      clf = DecisionTreeClassifier(max_depth=5)
      clf.fit(vanilla_gan_x, vanilla_gan_y)
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
      results[name][i+1]["vanilla_gan"]['dt'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]


      clf = GaussianNB()
      clf.fit(vanilla_gan_x, vanilla_gan_y)
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
      results[name][i+1]["vanilla_gan"]['nb'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]


      clf = LogisticRegression()
      clf.fit(vanilla_gan_x, vanilla_gan_y)
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
      results[name][i+1]["vanilla_gan"]['lr'] = [score, precision, recall, f1, auc, tp, fp, tn, fn, weighted_precision, weighted_recall, weighted_f1]
      append_record(results)

dict1 = results
results = {}

nan = 0
for nam in dict1 :
    if(len(dict1[nam]) < 10) :
        print("there are not 10 iterations for " + nam + ", hence, results will not be calculated for it")
        continue
    results[nam] = {"vanilla_gan":{}}
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
      final_results[name] = {"vanilla_gan":{}}
      for typ in results[name] :
            for classifier in results[name][typ] :
                  final_results[name][typ][classifier] = []
                  for values in results[name][typ][classifier] :
                        final_results[name][typ][classifier].append(sum(values)/len(values))

with open('./results/averaged_results_vanilla_gan.txt', 'w') as f:
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