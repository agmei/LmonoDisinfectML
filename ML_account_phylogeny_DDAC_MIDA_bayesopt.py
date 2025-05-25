#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 08:17:51 2023

ML model training for DDAC & MIDA using the best performing models form BenchmarkDR.
Also accounting for phylogenetic groups.

@author: algm
"""

import pandas as pd
import argparse
import numpy as np
import sys
import yaml
import os
import time
from joblib import dump, load
from sklearn.model_selection import cross_validate, LeaveOneGroupOut, LeavePGroupsOut, GroupKFold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier,  AdaBoostClassifier
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV


def make_scoring_dict(method, model, params, cv_result, tdelta):
        eval_dict=dict()
        
        eval_dict.setdefault('Method', model)
        eval_dict.setdefault('Parameters', dict(params))
        eval_dict.setdefault('Time (hh:mm:ss)', time.strftime("%H:%M:%S", time.gmtime(tdelta)))
        
        if method == 'regression':
            eval_dict.setdefault('mean_squared_error', round(abs(cv_result['test_neg_mean_squared_error']).mean(),4))
            eval_dict.setdefault('mean_squared_log_error', round(abs(cv_result['test_neg_mean_squared_log_error']).mean(),4))
            eval_dict.setdefault('r2', round(cv_result['test_r2'].mean(),4))
        
        if method == 'classification':
            eval_dict.setdefault('accuracy', round(cv_result['test_accuracy'].mean(),4))
            eval_dict.setdefault('balanced_accuracy', round(cv_result['test_balanced_accuracy'].mean(),4))
            eval_dict.setdefault('f1_weighted', round(cv_result['test_f1_weighted'].mean(),4))
        
        return eval_dict


####   PARSE ARGUMENT   ####

parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('-f', '--features', action='store', dest='feats',
                    help='''Path to feature matrix''')
parser.add_argument('-l', '--lables', action='store', dest='labels',
                    help='''Path to label matrix''')
parser.add_argument('-b', '--biocide', action='store', dest='bioc',
                    help='''Biocide name short cut used in feature and label matrix (e.g., 'BC')''')
parser.add_argument('-m', '--method', action='store', dest='method',
                    help='''Method of the ML training (e.g., regression, classification)''')
parser.add_argument('-g', '--groups', action='store', dest='groups',
                    help='''Value to select the phylogenetic groups for ML training (i.e., cc, mlst, kma, poppunk)''')
parser.add_argument('-o', '--outdir', action='store', dest='outd',
                    help='''Directory where to store the output''')

args = parser.parse_args()



# store arguments
features_f=os.path.abspath(args.feats)
labels_f=os.path.abspath(args.labels)
biocide=args.bioc
outd=os.path.abspath(args.outd)
method=args.method
group_l=args.groups

print(' '.join(sys.argv) + '\n')


# define variables
features=pd.read_csv(features_f, index_col=0)
labels=pd.read_csv(labels_f, index_col=0)


fitmodeld = os.path.join(outd,'fitted_models')
cv_resd = os.path.join(outd,'cv_objects')

if not os.path.isdir(outd):
    os.makedirs(outd)

if not os.path.exists(fitmodeld):
    os.makedirs(fitmodeld)

if not os.path.exists(cv_resd):
    os.makedirs(cv_resd)

# fix sample name DTU2015_1176_PRJ12345_listeria_subsp._15120858_03_R1 in features matrix
# to fit to labels
features.rename(index={"DTU2015_1176_PRJ12345_listeria_subsp":"DTU2015_1176_PRJ12345_listeria_subsp._15120858_03"}, inplace=True)

# join matrix
matrix=labels.join(features)

# dropping this sample cause it should not be in the matrix
if "DTU_2022_1016846_1_SI_SKB110_S128" in matrix.index:
    print('We found DTU_2022_1016846_1_SI_SKB110_S12')
    matrix.drop("DTU_2022_1016846_1_SI_SKB110_S128", inplace=True)
    labels.drop("DTU_2022_1016846_1_SI_SKB110_S128", inplace=True)


if matrix.shape[0] != labels.shape[0]:
    print(f"!!! Caution matrix length changed !!! labels: {labels.shape}, features: {features.shape}, comb matrix: {matrix.shape}")
    


X=matrix.drop(biocide, axis=1)/100
y=matrix[biocide]
print(f"Input matrix shape: {X.shape}")


#import new phylogeny file and choose which groups. Also maybe save the factorization of the classes. 
phylo_g=pd.read_csv(os.path.realpath("data/clusters/disinf_isolates_clusters_file.csv"), names=['id','cc', 'mlst', 'kmer_95', 'poppunk' ], header=0, index_col=0)


# subset groups to amount of samples
if biocide == "DDAC" or biocide == "MIDA":
    phylo_g=phylo_g.loc[labels.index]

groups=pd.factorize(phylo_g[group_l])[0]

# sys.exit()

#%%

# define parameter search spaces
param_grid={'regression':
  {'sklearn_LinR_l1':
    {'model': Lasso(),
    'params':
      {'alpha': Real(0.5, 2, prior='log-uniform')}
    },

  'sklearn_GBTR':
    {'model': GradientBoostingRegressor(),
    'params':
      {'n_iter_no_change': Categorical([20]),
      'learning_rate': Real(1e-3, 1e+1, prior='log-uniform'),
      'n_estimators': Integer(100, 1000),
      'min_samples_split': Integer(2, 15),
      'max_depth': Integer(5, 30)
      }
    },

  'sklearn_RFR':
    {'model': RandomForestRegressor(),
    'params':
      {'n_estimators': Integer(50,500),
      'max_depth': Integer(5, 30),
      'min_samples_split': Integer(2, 15),
      'min_samples_leaf': Integer(1, 10)
      }
    }
  },

  'classification':
  {'sklearn_RFC':
    {'model': RandomForestClassifier(),
    'params':
      {'class_weight': Categorical(['balanced']),
      'n_estimators': Integer(50,500),
      'max_depth': Integer(5, 30),
      'min_samples_split': Integer(2, 15),
      'min_samples_leaf': Integer(1, 10)
      }
    },

  'sklearn_GBTR':
    {'model': GradientBoostingClassifier(),
    'params':
      {'max_features': Categorical([1]),
      'n_iter_no_change': Categorical([20]),
      'learning_rate': Real(1e-3, 1e+1, prior='log-uniform'),
      'n_estimators': Integer(100, 1000),
      'min_samples_split': Integer(2, 15),
      'max_depth': Integer(5, 30)
      }
    },

  'sklearn_LR_l1':
    {'model': LogisticRegression(),
    'params':
      {'penalty': Categorical(['l1']),
      'class_weight': Categorical(['balanced']),
      'max_iter': Categorical([100000]),
      'solver': Categorical(['saga']),
      'C': Real(1e-4, 1e+1, prior='log-uniform')
      }
     },
      
   'sklearn_ADB':
     {'model':  AdaBoostClassifier(),
      'params':
        {'n_estimators': Integer(50,500),
        'learning_rate': Real(1e-4, 1e+1, prior='log-uniform'),
        }
     },
         
   'sklearn_SVM_l1':
      {'model': LinearSVC(),
      'params':
        {'penalty': Categorical(['l1']),
        'class_weight': Categorical(['balanced']),
        'dual': Categorical([False]),
        'max_iter': Categorical([1000000]),
        'loss': Categorical(['squared_hinge']),
        'C': Real(1e-4, 1e+1, prior='log-uniform')
        }
      }
  }
}


# adapted from https://inria.github.io/scikit-learn-mooc/python_scripts/cross_validation_nested.html
# Cross_validate does not have the group function but you can pass down the groups parameter through fit_params.
# adapted from https://stackoverflow.com/questions/60996995/use-groupkfold-in-nested-cross-validation-using-sklearn answer 4

inner_cv = GroupKFold(4)
outer_cv = GroupKFold(5)


if method == 'regression':
    score_hyper='neg_mean_squared_error'
    scores = ['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error']
    

if method == 'classification':
    score_hyper='balanced_accuracy'
    # original scores from BenchmarkDR, I am just not sure if it makes a lot of sense for some of the scores 
    # e.g., roc_auc, tp etc. if there is just one sample in the test set...
    # scores = ['accuracy','balanced_accuracy','f1_score','roc_auc','tp','tn','fp','fn','sensitivity','specificity']
    scores = ['accuracy','balanced_accuracy','f1_weighted']


summary_results=list()
bayes_iter=30

for model, values in param_grid[method].items():
    tstart=time.time()
    
    tmp_model=values['model']
    tmp_params=values['params']
    
    # Inner cross-validation for parameter search
    hyperopt = BayesSearchCV(tmp_model, tmp_params, random_state=3, n_iter=bayes_iter, verbose=1, cv=inner_cv, n_jobs=1, scoring=score_hyper)
    
    print(f"Working on: {tmp_model}")
    
    # Outer cross-validation to compute the testing score
    cv_result = cross_validate(hyperopt, X, y, cv=outer_cv, groups=groups, fit_params={'groups': groups}, verbose=1, n_jobs=-1, scoring=scores, return_train_score=True, return_estimator=True)
    print(
        f"The mean {score_hyper} score using nested cross-validation is: "
        f"{cv_result[f'test_{score_hyper}'].mean():.3f} ± {cv_result[f'test_{score_hyper}'].std():.3f}"
        '\n'
    )
    
    # extracting the best parameters from the cross-validation
    best_score=abs(cv_result[f'test_{score_hyper}']).min()
    best_score_index=np.where(abs(cv_result[f'test_{score_hyper}']) == best_score)[0][0]
    best_params=cv_result['estimator'][best_score_index].best_params_
    
    # just to be safe, refit the best estimator on the entire data. 
    fitted_model=cv_result['estimator'][best_score_index].best_estimator_.fit(X, y)
    
    
    tdelta=(time.time()-tstart)
    summary_results.append(make_scoring_dict(method, model, best_params, cv_result, tdelta))
    
    # save fitted model
    
    dump(fitted_model, os.path.join(fitmodeld ,f'{model}_phylo_fitted.joblib'))
    
    # save cv_results object
    dump(cv_result, os.path.join(cv_resd ,f'{model}_cv_result.joblib'))
    


# push summary_rsults in a pandas and then to csv

summary_res_df=pd.DataFrame(summary_results).set_index('Method')
summary_res_df.to_csv(os.path.join(outd, f'{method}_summary_results.csv'))

