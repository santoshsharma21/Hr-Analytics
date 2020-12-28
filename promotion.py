import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def target_encoding(train, test, target, cols):
    train_copy = train.copy()
    train_copy['target'] = target
    
    skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 21)
    # loop
    for train_index, val_index in skf.split(train, target):
        train_fold, val_fold = train.iloc[train_index], train.iloc[val_index]
        train_y = target[train_index]
        train_fold['target'] = train_y

        for col in cols:
            train.loc[val_index, col+'_target_encoded'] = val_fold[col].map(train_fold.groupby(col).target.mean())
    # for test set
    for col in cols:
        test[col +'_target_encoded'] = test[col].map(train_copy.groupby(col).target.mean())
    return train, test

# Initialize model
def run_xgb(train_fold, val_fold, y, test):
    xgb = XGBClassifier(
                        n_estimators = 1841,
                        max_depth = 4,
                        learning_rate =  0.01482763573605118, 
                        min_child_weight = 0.6045387236275916, 
                        subsample = 0.960107932565399, 
                        colsample_bytree = 0.7983133043770743,
                        scale_pos_weight = 3,
                        random_state = 21,
                        n_jobs = -1
                        )
    xgb.fit(train_fold, y)
    val_pred_label = xgb.predict(val_fold)
    val_pred_prob = xgb.predict_proba(val_fold)[:,1]
    test_pred = xgb.predict_proba(test)[:,1]
    
    return val_pred_label, val_pred_prob, test_pred

# LightGBM
def run_lgb(train_fold, val_fold, y, test):
    lgb = LGBMClassifier(
                         objective = 'binary',
                         num_leaves = 665, 
                         max_depth = 11, 
                         learning_rate = 0.011925312020895437, 
                         n_estimators = 1191,
                         min_child_weight = 0.7564376146047985, 
                         min_child_samples = 277, 
                         subsample = 0.6080548207494728, 
                         colsample_bytree = 0.7879049095473108,
                         max_bin = 181, 
                         scale_pos_weight = 2, 
                         reg_alpha = 0.00017606426191632533, 
                         reg_lambda = 2.984547229148173,
                         random_state = 21,
                         n_jobs = -1
                        )
    lgb.fit(train_fold, y)
    val_pred_label = lgb.predict(val_fold)
    val_pred_prob = lgb.predict_proba(val_fold)[:,1]
    test_pred = lgb.predict_proba(test)[:,1]
    
    return val_pred_label, val_pred_prob, test_pred                   
    
# Catboost
def run_cat(train_fold, val_fold, y, test, catfea):
    catb = CatBoostClassifier(
                             iterations = 823, 
                             depth = 7, 
                             learning_rate = 0.010885582797311217, 
                             random_strength = 0.6024649278875751, 
                             l2_leaf_reg =  0.89274890886717, 
                             bagging_temperature = 0.9323437952247747, 
                             scale_pos_weight = 3,
                             verbose = 0, 
                             cat_features = catfea,
                             random_state = 21,
                            )
    catb.fit(train_fold, y)
    val_pred_label = catb.predict(val_fold)
    val_pred_prob = catb.predict_proba(val_fold)[:,1]
    test_pred = catb.predict_proba(test)[:,1]
    
    return val_pred_label, val_pred_prob, test_pred

if __name__ == "__main__":
    # read data
    train = pd.read_csv('train.csv')
    test_x = pd.read_csv('test.csv')
    sub = pd.read_csv('sample_submission.csv')
    train_x = train.drop(['is_promoted'], axis = 1)
    target = train['is_promoted'].values
    
    # Print shape of data
    print('Train shape', train.shape)
    print('Test shape', test_x.shape)

    # impute missing value
    print('Imputing missing value....')
    train_x['education'] = train_x['education'].fillna(train_x['education'].mode()[0])
    train_x['previous_year_rating'] = train_x['previous_year_rating'].fillna(0)
    #Test
    test_x['education'] = test_x['education'].fillna(test_x['education'].mode()[0])
    test_x['previous_year_rating'] = test_x['previous_year_rating'].fillna(0)
    
    # creating new variables
    print('creating new features....') 
    # avg training score by department
    avg_score = train_x.groupby('department')['avg_training_score'].mean().to_dict()
    train_x['avg_training_score_by_dept'] = train_x['department'].map(avg_score)
    test_x['avg_training_score_by_dept'] = test_x['department'].map(avg_score)
    
    # avg previous year rating by department
    avg_rating = train_x.groupby('department')['previous_year_rating'].mean().to_dict()
    train_x['avg_rating_score_by_dept'] = train_x['department'].map(avg_rating)
    test_x['avg_rating_score_by_dept'] = test_x['department'].map(avg_rating)
    
    # count of dept by rating
    gdf = train_x.groupby(['department','previous_year_rating'])['department'].count().reset_index(name = 'count')
    gdf.columns = ['department','previous_year_rating','count_of_dept_by_rating']
    train_x = pd.merge(train_x, gdf, on = ['department','previous_year_rating'], how = 'left')
    test_x = pd.merge(test_x, gdf, on = ['department','previous_year_rating'], how = 'left')
    
    # target encoding
    cols = ['department','region']
    train_x, test_x = target_encoding(train_x, test_x, target, cols)
    
    # label encoding
    cols_for_label_encoding = ['education','gender','recruitment_channel']
    for col in cols_for_label_encoding:
        encoder = LabelEncoder()
        encoder.fit(train_x[col].values.tolist() + test_x[col].values.tolist())
        train_x.loc[:,col] = encoder.transform(train_x[col].values.tolist())
        test_x.loc[:,col] = encoder.transform(test_x[col].values.tolist())

    print('Data Prep completed....')
    
    # droping columns
    print('dropping columns')
    cols_to_drop = ['employee_id','department','region']
    train_x.drop(cols_to_drop, axis = 1, inplace = True)
    test_x.drop(cols_to_drop, axis = 1, inplace = True)
    
    print('Training model.....')

    fold = 11
    oof_pred = np.zeros(len(train_x))
    test_pred_xgb = 0
    test_pred_lgb = 0
    test_pred_cat = 0
     
    skf = StratifiedKFold(n_splits = fold, shuffle = True, random_state = 42)
    for i, (t_, v_) in enumerate(skf.split(train_x, target), 1):
        xtrain, xval = train_x.iloc[t_,:], train_x.iloc[v_,:]
        ytrain, yval = target[t_], target[v_]
        print(f"Training model on Fold {i} of fold {fold}")
        # run xgboost
        xgb_val_pred_label, xgb_val_pred_prob, xgb_test_pred = run_xgb(xtrain.values, xval.values, ytrain, test_x.values) 
        test_pred_xgb += xgb_test_pred

        # run lightgbm
        lgb_val_pred_label, lgb_val_pred_prob, lgb_test_pred = run_lgb(xtrain.values, xval.values, ytrain, test_x.values) 
        test_pred_lgb += lgb_test_pred

        # run catboost
        catf = ['education','gender','no_of_trainings','recruitment_channel']
        cat_val_pred_label, cat_val_pred_prob, cat_test_pred = run_cat(xtrain, xval, ytrain, test_x, catf) 
        test_pred_cat += cat_test_pred
        
         # out of fold prediction
        validation_pred = (xgb_val_pred_prob + lgb_val_pred_prob + cat_val_pred_prob)/3
        oof_pred[v_] = validation_pred

        print(f"F1 score on validation data using XGB = {f1_score(yval, xgb_val_pred_label)}")
        print(f"F1 score on validation data using LGBM = {f1_score(yval, lgb_val_pred_label)}")
        print(f"F1 score on validation data using CAT = {f1_score(yval, cat_val_pred_label)}")
        print(f"F1 score on validation data using (XGB + LGBM + CAT) = {f1_score(yval, np.where(validation_pred >= 0.5,1,0))}\n")
    
    test_pred_xgb = (test_pred_xgb/fold)
    test_pred_lgb = (test_pred_lgb/fold)
    test_pred_cat = (test_pred_cat/fold)
    final_test_pred = (test_pred_xgb + test_pred_lgb + test_pred_cat)/3

    print(f"OOF F1 score {f1_score(target, np.where(oof_pred >= 0.5, 1,0))}")
    print(f"Model training completed.........")

    # write submission file
    sub['is_promoted'] = np.where(final_test_pred >= 0.5, 1,0)
    sub.to_csv('xgb_lgb_cat_11fold.csv', index = False)