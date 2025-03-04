import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, f1_score,roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import textwrap


# Read in Data, Create X and y
X = pd.read_csv('FM_2800.csv', index_col=0)
# X = X.iloc[:,12:] ## For notes only
y_data_pre = pd.read_csv('data_2800.csv')
y_data = y_data_pre[['Unnamed: 0', 'patientid', 'annot']]
y = y_data_pre['annot']


Nboot = 10000
random_seed = 2023

# Intializations
all_predictions = []
all_true_labels = []
all_row_numbers = []
auc_cv = []
auc_pr = []
f1_cv = []
cf_cv = []
predictions = []
roc_curves = []
pr_curves = []
final_tree_num = []
final_max_depth = []
final_prune_strength = []
final_min_leaf_samp =[]


kf = KFold(n_splits = 5, shuffle = True, random_state = random_seed)

for train_index, test_index in kf.split(X):
    Xtr, Xte = X.loc[train_index], X.loc[test_index]
    ytr, yte = y.loc[train_index], y.loc[test_index]

    model = RandomForestClassifier(
        max_features='sqrt', 
        random_state=random_seed)

    # hyperparameters
    search_spaces = {
        'n_estimators': (50,150),
        'max_depth': (2,7),
        'ccp_alpha': (1e-4, 1e-1),
        'min_samples_leaf': (10,75),
    }
    
    model_cv = BayesSearchCV(model,
        search_spaces,
        n_iter=50, scoring='roc_auc', n_jobs=7,
        cv=5, random_state=random_seed)
    
    # for column in Xtr.columns:
    #     Xtr[column] = Xtr[column].astype(int)

    # ytr = ytr.astype(int)

    model_cv.fit(Xtr,ytr)   

    # Find Best hyperparameters
    best_hparams = model_cv.best_params_
    best_tree_num = best_hparams['n_estimators']
    best_max_depth = best_hparams['max_depth']
    best_prune_strength = best_hparams['ccp_alpha']
    best_min_leaf_samp = best_hparams['min_samples_leaf']
    final_tree_num.append(best_tree_num)
    final_max_depth.append(best_max_depth)
    final_prune_strength.append(best_prune_strength)
    final_min_leaf_samp.append(best_min_leaf_samp)


    model = model_cv.best_estimator_
    yte_pred = model.predict_proba(Xte)[:,1]

    fpr, tpr, cutoffs = roc_curve(yte, yte_pred)
    best_cutoff = cutoffs[np.argmax(tpr - fpr)]
    yte_pred_bin =(yte_pred>best_cutoff).astype(int)

    auc_cv.append(roc_auc_score(yte, yte_pred))
    f1_cv.append( f1_score(yte, yte_pred_bin) )
    cf_cv.append( confusion_matrix(yte, yte_pred_bin) )
    predictions.append(yte_pred_bin)

    # Store the predictions and true labels for this fold
    all_predictions.extend(yte_pred_bin)
    all_true_labels.extend(yte)
    all_row_numbers.extend(y_data.iloc[test_index]['Unnamed: 0'])

    # Save info for the plots
    # roc_curves.append((fpr, tpr, roc_auc_score(yte, yte_pred)))
    # precision, recall, thresholds = precision_recall_curve(yte, yte_pred)
    # pr_curves.append((recall, precision, auc(recall, precision)))
    # auc_pr_loop = auc(recall, precision)
    # auc_pr.append(auc_pr_loop)

    precision, recall, thresholds = precision_recall_curve(yte, yte_pred)
    auc_pr_loop = auc(recall, precision)
    auc_pr.append(auc_pr_loop)

    yte = yte.reset_index(drop=True)

    # Perform bootstrapping iterations
    for i in tqdm(range(Nboot)):
        index = np.random.randint(0, len(yte),len(yte))
        sample_te = yte[index]
        sample_te_pred = yte_pred[index]
        fpr, tpr, cutoffs = roc_curve(sample_te, sample_te_pred,drop_intermediate=False)
        roc_curves.append((fpr, tpr, roc_auc_score(sample_te, sample_te_pred)))

        precision_boot, recall_boot, thresholds = precision_recall_curve(sample_te, sample_te_pred)
        pr_curves.append((recall_boot, precision_boot, auc(recall_boot, precision_boot)))

    print(auc_cv)
    print(auc_pr)


#####################################
# AUC_ROC CURVE
#####################################

# Calc max length
max_fpr_length = max([len(fpr) for fpr, _, _ in roc_curves])

# Pad fpr and tpr arrays to match the max length
padded_fprs = []
padded_tprs = []
for fpr, tpr, _ in roc_curves:
    padded_fpr = np.concatenate([fpr, [fpr[-1]] * (max_fpr_length - len(fpr))])
    padded_tpr = np.concatenate([tpr, [tpr[-1]] * (max_fpr_length - len(tpr))])
    padded_fprs.append(padded_fpr)
    padded_tprs.append(padded_tpr)

# calc mean fpr, tpr
mean_fpr = np.mean(padded_fprs, axis=0)
mean_tpr = np.mean(padded_tprs, axis=0)
mean_auc = np.mean([auc_score for _, _, auc_score in roc_curves])
youden = np.max(padded_tpr-padded_fpr)
max_index = np.argmax(padded_tpr - padded_fpr)
max_tpr = padded_tpr[max_index]
max_fpr = padded_fpr[max_index]

# Calc percentiles for the middle 95% of bootstrapped tpr arrays
percentile_2_5 = np.percentile(padded_tprs, 2.5, axis=0)
percentile_97_5 = np.percentile(padded_tprs, 97.5, axis=0)
upper_auc = auc(mean_fpr, percentile_97_5)
lower_auc = auc(mean_fpr, percentile_2_5)

# Plot the middle 95% shaded area
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) #
plt.fill_between(mean_fpr, percentile_2_5, percentile_97_5, color='gray', alpha=0.3, label='Middle 95% Bootstapped')

# Diagonal line representing random classifier
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate (FPR)', fontsize = 12)
plt.ylabel('True Positive Rate (TPR)', fontsize = 12)

# Add mean AUC-ROC value to legend
plt.plot(mean_fpr, mean_tpr, color='red', label='Mean ROC Curve')
plt.legend(loc='lower right', title=f"AUROC = {mean_auc:.3f} ({upper_auc:.3f} - {lower_auc:.3f})")
plt.title('(A)', fontsize = 20, loc='left')


#####################################
# PR CURVE
#####################################

# Calc max length
max_recall_length = max([len(recall) for recall, _, _ in pr_curves])

# Pad recall and precision arrays to match the max length
padded_recalls = []
padded_precisions = []
for recall, precision, _ in pr_curves:
    padded_recall = np.concatenate([recall, [recall[-1]] * (max_recall_length - len(recall))])
    padded_precision = np.concatenate([precision, [precision[-1]] * (max_recall_length - len(precision))])
    padded_recalls.append(padded_recall)
    padded_precisions.append(padded_precision)

# calc mean recall, precision
mean_recall = np.mean(padded_recalls, axis=0)
mean_precision = np.mean(padded_precisions, axis=0)
mean_auc = np.mean([auc_score for _, _, auc_score in pr_curves])
youden_pr = np.max(padded_recall- padded_precision)
max_index_pr = np.argmax(padded_recall + padded_precision)
max_rec_pr = padded_recall[max_index_pr]
max_pre_pr = padded_precision[max_index_pr]

# Calculate percentiles for the middle 95% of bootstrapped recall and precision arrays
percentile_2_5 = np.percentile(padded_precisions, 2.5, axis=0)
percentile_97_5 = np.percentile(padded_precisions, 97.5, axis=0)
upper_auc = auc(mean_recall, percentile_97_5)
lower_auc = auc(mean_recall, percentile_2_5)

# Plot the middle 95% shaded area
plt.subplot(1,2,2)
plt.fill_between(mean_recall, percentile_2_5, percentile_97_5, color='gray', alpha=0.3, label='Middle 95% Bootstrapped')

# Plot mean PR curve as a red line
plt.plot(mean_recall, mean_precision, color='red', label='Mean PR Curve')

# Add mean AUC-PR value, upper/lower AUC values to legend
plt.legend(loc='lower right', title=f"AUPRC = {mean_auc:.3f} ({upper_auc:.3f} - {lower_auc:.3f})")
plt.xlabel('Recall', fontsize = 12)
plt.ylabel('Precision', fontsize = 12)
plt.title('(B)', fontsize = 20, loc='left')
plt.tight_layout()
plt.savefig('RF.png')
plt.show()


# Create a DataFrame to hold the results
results_df = pd.DataFrame({'Unnamed: 0': all_row_numbers, 'annot': all_true_labels, 'prediction': all_predictions})

# Save the DataFrame to a CSV file
# results_df.to_csv('/home/ubuntu/PROJECTS/CHF/model/Data/RF_iter1-2.csv', index=False)

# Calculate the final AUC and F1
auc_final = np.mean(auc_cv)
f1_final = np.mean(f1_cv)

# Save all the data
df = pd.DataFrame()
df['auc'] = auc_cv
df['f1s'] = f1_cv

df['tree_num'] = final_tree_num
df['max_depth'] = final_max_depth
df['prune_strength'] = final_prune_strength
df['min_leaf_samp'] = final_min_leaf_samp

df.to_csv('RF_results.csv')






