# Genrative Adversarial Based oversampling of Software fault prediction

We have created nine different script files, each belonging to an individual technique to handle the imbalance in the data.

We have used nine methods to hanlde this imbalance :
  1. WGANGP based oversampling
  2. SMOTE oversampling 
  3. ADASYN oversampling
  4. Random oversampling  
  5. Random undersampling
  6. Borderline SMOTE
  7. Adaboost
  8. Vaniila GAN based Oeversampling
  9. CTGAN based oversampling

One can choose any technique from above and accordingly, he/she can run the corresponding script file to generate balanced datasets and generating the performace report 
with various performace metric.

We are generating below performace metrics :
  1. Accuracy
  2. Precision
  3. Recall
  4. F1_score
  5. AUC 
  6. TP
  7. FP 
  8. TN 
  9. FN

Among above mentioned metrics, precision, recall, f1_score and AUC are most reliable in case of imbalanced data.
We are saving the results of these techniques in directory results.

Procedure :
  At first, we are splitting the imbalanced data in two sets, training with 70% data samples and test with rest of the data so that we will have real testing data.
  Now, using one of the techniques mentioned above and training data, we generate balanced data. Thereafter we train classifiers over this balanced training data.
  And than, using test data and trained classiffer, we generate all the metrics mentioned above.
  We repeat this process 10 times for each technique and than we take average of all results to get a better insight of performace.

  We have used five different classifiers for classification :
    1. K-nearest neighbors
    2. Random Forest
    3. Decision Tree
    4. Naive Bayes
    5. Logistic Regression
