#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:09:16 2025

@author: sai
"""
print("\n\n********** Modeling **********\n\n")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc

df = pd.read_csv("df_dummies.csv")


#%%

train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Is_like'])
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['Is_like'])

# train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42)
# test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42)

#%%
full_counts = df['Is_like'].value_counts()
train_counts = train['Is_like'].value_counts()
TandV_counts = test_and_validate['Is_like'].value_counts()
test_counts  = test['Is_like'].value_counts()
val_counts   = validate['Is_like'].value_counts()

# print(f'Distribution of liked songs before split: {full_counts}')
# print(f'Distribution of liked songs in training set: {train_counts}')
# print(f'Distribution of liked songs in test and validate set: {TandV_counts}')
# print(f'Distribution of liked songs in test set: {test_counts}')
# print(f'Distribution of liked songs in validate set: {val_counts}')

## making a table to check the distribution of each dataset
table = pd.DataFrame({
    'Full': full_counts,
    'Train': train_counts,
    'Test and Validate': TandV_counts,
    'Test': test_counts,
    'Validate': val_counts
})

# Add a row showing the proportion of 1s
table.loc['Proportion_1'] = table.loc[1] / table.sum()

table.style \
    .format('{:.0f}', subset=pd.IndexSlice[[0, 1], :]) \
    .format('{:.2f}', subset=pd.IndexSlice[['Proportion_1'], :])

#%%

X_train = train.drop(['Is_like'], axis = 1)

Y_train = train['Is_like']

X_test = test.drop(['Is_like'], axis = 1)

Y_test = test['Is_like']

X_val = validate.drop(['Is_like'], axis = 1)

Y_val = validate['Is_like']
#%%
"""
XGBoost classifier - 
objective: binary:logistic for binary classification
Eval_metric: Area Under Curve - it's about ranking and good for imbalance data
"""
#%%
from xgboost import XGBClassifier

model = XGBClassifier(objective='binary:logistic', eval_metric='auc', n_estimators=300)
model.fit(X_train, Y_train)
#%%
"""
First Guess - we use test- sample 
"""
#%%
y_pred = model.predict(X_test) # 
y_pred_proba_val = model.predict_proba(X_test)[:, 1]
# y_pred_proba_val = model.predict_proba(X_test)
#print(y_pred_proba_val)


#%%
print("\n--- Validation Scores (Base model) ---")
print(f"Accuracy by default:    {accuracy_score(Y_test, y_pred):.4f}")
print(f"F1 Score:               {f1_score(Y_test, y_pred):.4f}")
print(f"AUC:                    {roc_auc_score(Y_test, y_pred_proba_val):.4f}\n\n")

#%%

"""
*******Hyperparameter Tuning********
We use validation set for hyper-parameter tuning
"""

#%%


param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 5, 10] 
}

xgb = XGBClassifier(objective='binary:logistic', eval_metric='auc')


random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1', # Optimize for AUC (or use 'f1' or 'accuracy')
    cv=5,              # 5-fold Cross-Validation
    verbose=1,
    random_state=42,
    n_jobs=-1          # Use all available cores
)

print("Starting Hyperparameter Tuning...")
random_search.fit(X_train, Y_train)

# 5. Get the best results
print(f"\nBest Parameters: {random_search.best_params_}")
print(f"Best Training AUC Score: {random_search.best_score_:.4f}")

# 6. Evaluate on Validation Set with best model
best_model = random_search.best_estimator_

y_pred_val = best_model.predict(X_val)
y_pred_proba_val = best_model.predict_proba(X_val)[:, 1]

print("\n--- Validation Scores (Base model) ---")
print(f"Accuracy by default:    {accuracy_score(Y_val, y_pred_val):.4f}")
print(f"F1 Score:               {f1_score(Y_val, y_pred_val):.4f}")
print(f"AUC:                    {roc_auc_score(Y_val, y_pred_proba_val):.4f}\n\n")

#%%

matrix = confusion_matrix(y_pred_val, Y_val)
matrix

# %%
df_confusion = pd.DataFrame(matrix, index=['Like','Dislike'],columns=['Like','Dislike'])

df_confusion

plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Dislike (0)', 'Predicted Like (1)'],
            yticklabels=['Actual Dislike (0)', 'Actual Like (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - Validation Set')
plt.show()

# 4. Detailed Breakdown
TN, FP, FN, TP = matrix.ravel()
print("\n--- Breakdown ---")
print(f"True Negatives (Correctly identified Dislikes): {TN:.4f}")
print(f"False Positives (Predicted Like, actually Dislike): {FP:.4f}")
print(f"False Negatives (Predicted Dislike, actually Like): {FN:.4f}")
print(f"True Positives (Correctly identified Likes): {TP:.4f}")

# Sensitivity, hit rate, recall, or true positive rate
Sensitivity  = float(TP)/(TP+FN)*100
print(f"Sensitivity or TPR: {Sensitivity:.4f}%")  
print(f"There is a {Sensitivity:.4f}% chance of correctly predicting if user likes song\n")


# %%
# Specificity or true negative rate
Specificity  = float(TN)/(TN+FP)*100
print(f"Specificity or TNR: {Specificity:.4f}%") 
print(f"There is a {Specificity:.4f}% chance of predicting if user dislikes song\n")


# %%
# Precision or positive predictive value
Precision = float(TP)/(TP+FP)*100
print(f"Precision: {Precision:.4f}%")  
print(f"A user likes a song, and the probablity that is correct is {Precision:.4f}%\n")

# %%
# Negative predictive value
NPV = float(TN)/(TN+FN)*100
print(f"Negative Predictive Value: {NPV:.4f}%") 
print(f"User dislikes a song, but there is a {NPV:.4f}% chance that is incorrect\n" )

# %%
# Fall out or false positive rate
FPR = float(FP)/(FP+TN)*100
print( f"False Positive Rate: {FPR:.4f}%") 
print( f"There is a {FPR:.4f}% chance that predicting a user likes a song is incorrect\n")

# %%
# False negative rate
FNR = float(FN)/(TP+FN)*100
print(f"False Negative Rate: {FNR:.4f}%") 
print(f"There is a {FNR:.4f}% chance a prediciton of a user diliking a song is incorrect.\n")

# %%
# False discovery rate
FDR = float(FP)/(TP+FP)*100
print(f"False Discovery Rate: {FDR:.4f}%" )
print(f"Predicts a user likes a song, but there is a {FDR:.4f}% chance this is incorrect.\n")

# %% [markdown]
# ## Overall accuracy

# %%
ACC = float(TP+TN)/(TP+FP+FN+TN)*100
print(f"Accuracy: {ACC:.4f}%\n") 

# %%
print(f"Sensitivity or TPR: {Sensitivity:.4f}%")    
print(f"Specificity or TNR: {Specificity:.4f}%") 
print(f"Precision: {Precision:.4f}%")   
print(f"Negative Predictive Value: {NPV:.4f}%")  
print(f"False Positive Rate: {FPR:.4f}%") 
print(f"False Negative Rate: {FNR:.4f}%")  
print(f"False Discovery Rate: {FDR:.4f}%" )
print(f"Accuracy: {ACC:.4f}%") 

# %%
AUC = roc_auc_score(Y_val, y_pred_proba_val)
print("Validation AUC", AUC)

#%%

# write the values to a file append the writes for subsequent runs
df_write = pd.DataFrame([[Sensitivity, Specificity, Precision, NPV, FPR, FNR, FDR, ACC, AUC]])
df_write.to_csv('ProjectStats.csv', index=False, header=False, mode='a')

#%%
fpr, tpr, thresholds = roc_curve(Y_val, y_pred_proba_val)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
ax2.set_ylabel('Threshold', color='r')
ax2.set_xlim([fpr[0], fpr[-1]])

plt.show()