import numpy as np
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm
import textwrap

X = pd.read_csv('FM_2800.csv', index_col=0)
y_data_pre = pd.read_csv('data_2800.csv')
y_data = y_data_pre[['Unnamed: 0', 'patientid', 'annot']]
y = y_data_pre['annot']

Nboot = 10000
random_seed = 2023

all_predictions = []
all_true_labels = []
all_row_numbers = []

# outer loop
auc_cv = []
auc_pr = []
f1_cv = []
cf_cv = []
final_Cs = []
final_l1 = []
predictions = []
roc_curves = []
pr_curves = []

kf = KFold(n_splits = 5, shuffle= True, random_state = random_seed)

for train_index, test_index in kf.split(y_data_pre):
    Xtr, Xte = X.loc[train_index], X.loc[test_index]
    ytr, yte = y.loc[train_index], y.loc[test_index]

    
    model = LogisticRegression(
            penalty='elasticnet',
            class_weight=None, random_state=random_seed,
            solver='saga', max_iter=1000)
    
    search_spaces = {
         'C': (1e-2, 1e+2, 'log-uniform'),
         'l1_ratio': (0.01, 0.99),
    }
    model_cv = BayesSearchCV(model,
            search_spaces,
            n_iter=10, scoring='roc_auc', n_jobs=7,
            cv=5, random_state=random_seed)
    model_cv.fit(Xtr, ytr)

    #Find the Best hyperparameters and append them
    best_hparams = model_cv.best_params_
    best_C = best_hparams['C']
    best_l1_ratio = best_hparams['l1_ratio']
    final_Cs.append(best_C)
    final_l1.append(best_l1_ratio)

    model = model_cv.best_estimator_
    yte_pred = model.predict_proba(Xte)[:,1]    # Xte is testing features

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

    # store pr data
    precision, recall, thresholds = precision_recall_curve(yte, yte_pred)
    auc_pr_loop = auc(recall, precision)
    auc_pr.append(auc_pr_loop)

    yte = yte.reset_index(drop=True)

    coefficients = model.coef_[0]

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
plt.subplot(1,2,1)
plt.fill_between(mean_fpr, percentile_2_5, percentile_97_5, color='gray', alpha=0.3, label='95% CI')

# Diagonal line representing random classifier
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False positive rate', fontsize = 12)
plt.ylabel('True positive rate', fontsize = 12)

# Add mean AUC-ROC value to legend
plt.plot(mean_fpr, mean_tpr, color='red', label='Mean ROC curve')
plt.legend(loc='lower right', title=f"AUROC = {mean_auc:.3f} (95% CI {lower_auc:.3f} - {upper_auc:.3f})")
plt.title('(A)', fontsize = 20, loc='left')
#plt.savefig('AUC_ALL-2.png')
# plt.show()


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
plt.fill_between(mean_recall, percentile_2_5, percentile_97_5, color='gray', alpha=0.3, label='95% CI')

# Plot mean PR curve as a red line
plt.plot(mean_recall, mean_precision, color='red', label='Mean PR curve')

# Add mean AUC-PR value, upper/lower AUC values to legend
plt.legend(loc='lower right', title=f"AUPRC = {mean_auc:.3f} (95% CI {lower_auc:.3f} - {upper_auc:.3f})")
plt.xlabel('Recall', fontsize = 12)
plt.ylabel('Precision', fontsize = 12)
plt.title('(B)', fontsize = 20, loc='left')
plt.savefig('LR_PR_AUC.png')
plt.show()

# Calculate the optimal cutoff using the Youden's index
optimal_cutoff = cutoffs[np.argmax(tpr - fpr)]

# Apply the optimal cutoff to get binary predictions
final_predictions = (np.array(all_predictions) > optimal_cutoff).astype(int)

# Compute final recall and precision
final_recall = recall_score(all_true_labels, final_predictions)
final_precision = precision_score(all_true_labels, final_predictions)

# Get counts for recall and precision
tp_fn = sum(all_true_labels)  # Total positives (N for recall)
tp_fp = sum(final_predictions)  # Total predicted positives (N for precision)
tp = sum(np.array(all_true_labels) & final_predictions)  # True positives (n for recall and precision)

# Print results
print(f"Optimal Cutoff: {optimal_cutoff:.4f}")
print(f"Recall: {final_recall:.2%} ({tp}/{tp_fn})")
print(f"Precision: {final_precision:.2%} ({tp}/{tp_fp})")

# Create a DataFrame to hold the results
results_df = pd.DataFrame({'Unnamed: 0': all_row_numbers, 'annot': all_true_labels, 'prediction': all_predictions})

# Calculate the final AUC and F1
auc_final = np.mean(auc_cv)
f1_final = np.mean(f1_cv)

# Save all the data
df = pd.DataFrame()
df['auc'] = auc_cv
df['f1s'] = f1_cv
df['auc_pr'] = auc_pr
df['C'] = final_Cs
df['l1_ratio'] = final_l1
df['youden'] = youden
df['tpr'] = max_tpr
df['fpr'] = max_fpr

df_pred = pd.DataFrame()
df1 = pd.DataFrame(predictions[0])
df2 = pd.DataFrame(predictions[1])
df3 = pd.DataFrame(predictions[2])
df4 = pd.DataFrame(predictions[3])
df5 = pd.DataFrame(predictions[4])
df_pred = pd.concat([df1,df2,df3,df4,df5]).reset_index()

# Print final Data
print(f'AUCROC for ALL = {auc_final}')
print(f'final f1 for all = {f1_final}')
print(f' youden for all = {youden}')
print(f'tpr all = {max_tpr}')
print(f'fpr all = {max_fpr}')
print(f'recall all = {max_rec_pr}')
print(f'precision all = {max_pre_pr}')

df.to_csv('LR_res2800-2.csv')