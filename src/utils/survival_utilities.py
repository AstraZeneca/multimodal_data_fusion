# Multi-omics Pipeline Utilities

import numpy as np
import pandas as pd
import warnings
import eli5
from sksurv import datasets
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances


def hyperparameter_search(data: list, hyperparam_space: dict, model, splits: int=5):
    """
    Parameters
    ----------
    data: LIST, required
      data and labels to train the model and perform hyperparameter search
    hyperparam_space: DICTIONARY, required
      hyper parameter space to evaluate
    model: required
      type of model to fit
    splits: INTEGER, optional
      Number of splits to use in CV
      Defaults to 5
    """
    X, Y = data
    search = GridSearchCV(model, hyperparam_space, cv=splits)
    # Perform search
    result = search.fit(X, Y)
    return result


def prepare_data_sksurv(data_train, data_test, target_cols):#TODO: add handle_categorical as option
    """
    Method to prepare training & test sets to be used for training & evaluating scikit-survival models.
    Current version also converts categorical to numerical (ordinal) --TODO: Hard-coded; make dynamic.

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Training set (contains both feature & target columns)
    data_test: pd.DATAFRAME
      Test set (contains both feature & target columns)     
    
    Returns
    -------
    X_train: pd.DATAFRAME
      Training set features (columns:features, rows entries)
    Y_train: pd.DATAFRAME
      Training set targets; format (columns: duration & event, rows entries)
    X_test: pd.DATAFRAME
      Test set features (columns:features, rows entries)
    Y_test: pd.DATAFRAME
      Test set targets; format (columns: duration & event, rows entries)
    """
    
    verbose = 0 #TODO: Add as argument
    
    #Work on copies of the 2 datasets, so as to leave them unchanged:
    data_train_temp = data_train.copy()
    data_test_temp = data_test.copy()

    #Add a flag indicating if this is a train or a test example, before merging:
    data_train_temp['train'] = 1
    data_test_temp['train'] = 0
    data_full = pd.concat([data_train_temp, data_test_temp]) #Will only be used for converting categorical to numerical

    #Find categorical columns present in the data:
    #Removed 'age_at_initial_pathologic_diagnosis_clin' (already numerical), for interpretability 
    #cat_columns_list = ['ajcc_pathologic_tumor_stage_clin', 'gender_clin', 'race_clin', 'type_clin']

    cat_columns_list = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #Convert categorical to numerical:
        data_full_cat_to_num = data_full
        
#         #Option 1: 1-hot-encoding (could work best for XGB) -TODO: weird bug here...fix
#         for i in range(len(cat_columns_list)):
#             column_name = cat_columns_list[i]
#             if column_name in data_full.columns:
#                 df = pd.get_dummies(data_full, columns=[column_name])
#                 data_full_cat_to_num = pd.concat([data_full_cat_to_num, df], axis=1)
#                 data_full_cat_to_num.drop(column_name, axis=1, inplace=True)# Now that we have the dummy variables, we can get rid of the original variable      

        #Option 2: ordinal (could work best for CPH)
        if verbose == 1:
            print("Categorical features converted to numerical via ordinal encoding.")
        for i in range(len(cat_columns_list)):
            column_name = cat_columns_list[i]
            if column_name in data_full.columns:
                data_full_cat_to_num[column_name] = pd.Categorical(data_full_cat_to_num[column_name])
                mapping = dict(enumerate(data_full_cat_to_num[column_name].cat.categories))
                data_full_cat_to_num[column_name] = data_full_cat_to_num[column_name].cat.codes
                if verbose == 1:
                    print("Feature: "+column_name)
                    print("Mapping:")
                    print(mapping)
        
        #Separate train & test data again, getting rid of relevant flag column:
        data_train_temp = data_full_cat_to_num[data_full['train'] == 1]
        data_test_temp = data_full_cat_to_num[data_full['train'] == 0]
        
        X_train, Y_train = datasets.get_x_y(data_train_temp, attr_labels=target_cols, pos_label=1, survival=True)
        X_test, Y_test = datasets.get_x_y(data_test_temp, attr_labels=target_cols, pos_label=1, survival=True)
        
        X_train.drop('train', axis=1, inplace=True)
        X_test.drop('train', axis=1, inplace=True)  

        return X_train, Y_train, X_test, Y_test

def prepare_data_pycox(data_train, data_val, data_test):#TODO: add handle_categorical as option, add normalization type as option
    """
    Method to prepare training, test & evaluation sets to be used for training & evaluating pycox survival models.
    Current version also applies min-max scaling to numerical features and converts categorical to numerical (ordinal)
    --TODO: Hard-coded; make dynamic.

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Training set (contains both feature & target columns)
    data_val: pd.DATAFRAME
      Validation set (contains both feature & target columns)
    data_test: pd.DATAFRAME
      Test set (contains both feature & target columns)     
    
    Returns
    -------
    X_train: pd.DATAFRAME
      Training set features (columns:features, rows entries)
    Y_train: TUPLE
      Training set targets; format (duration, event)
    X_val: pd.DATAFRAME
      Validation set features (columns:features, rows entries)
    Y_val: TUPLE
      Validation set targets; format (duration, event)
    X_test: pd.DATAFRAME
      Test set features (columns:features, rows entries)
    Y_test: TUPLE
      Test set targets; format (duration, event)
    """
    verbose = 0 #TODO: Add as argument
    
    #Work on copies of the datasets so as to leave them unchanged:
    data_train_temp = data_train.copy()
    data_val_temp = data_val.copy()
    data_test_temp = data_test.copy()
    
    #1. Labels (event & duration); No label transformation needed for Deep Survival as it is continous: --TODO: if discrete models are added must also add label transforming for them 
    get_target = lambda df: (df['OS.time'].values, df['OS'].values)
    Y_train = get_target(data_train_temp)
    Y_val = get_target(data_val_temp)
    Y_test = get_target(data_test_temp)
    
    #Drop event & duration columns:
    data_train_temp = data_train_temp.drop(['OS.time', 'OS'], axis = 1)#, inplace = True
    data_val_temp = data_val_temp.drop(['OS.time', 'OS'], axis = 1)   
    data_test_temp = data_test_temp.drop(['OS.time', 'OS'], axis = 1)
    
    #2. Features
    cols_leave = []

    #Add a flag indicating if this is a training, test or validation example, before merging:
    data_train_temp['subset'] = 'train'
    data_test_temp['subset'] = 'test'
    data_val_temp['subset'] = 'val'
    data_full = pd.concat([data_train_temp, data_test_temp, data_val_temp]) #Will only be used for converting categorical to numerical
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #Convert categorical to numerical:
        
        data_full_cat_to_num = data_full   
        #Ordinal encoding of categorical features:
        if verbose == 1:
            print("Categorical features converted to numerical via ordinal encoding.")
        for i in range(len(cols_leave)):
            column_name = cols_leave[i]
            data_full_cat_to_num[column_name] = pd.Categorical(data_full_cat_to_num[column_name])
            mapping = dict(enumerate(data_full_cat_to_num[column_name].cat.categories))
            data_full_cat_to_num[column_name] = data_full_cat_to_num[column_name].cat.codes
            if verbose == 1:
                print("Feature: "+column_name)
                print("Mapping:")
                print(mapping)  
            
        #Separate train, validation & test data again, getting rid of relevant flag column:
        data_train_temp = data_full_cat_to_num[data_full['subset'] == 'train']
        data_test_temp = data_full_cat_to_num[data_full['subset'] == 'test']
        data_val_temp = data_full_cat_to_num[data_full['subset'] == 'val']
        
        data_train_temp = data_train_temp.drop(['subset'], axis = 1)#, inplace = True
        data_val_temp = data_val_temp.drop(['subset'], axis = 1)   
        data_test_temp = data_test_temp.drop(['subset'], axis = 1)
           
    cols_normalize = [item for item in data_train_temp.columns if item not in cols_leave]#Remaining columns; to be standardized
    
    #Choose normalization method for continuous columns: --TODO: Add as argument, probably allow for separate per modality
    #normalize = [([col], StandardScaler()) for col in cols_normalize]#Uses sklearn_pandas.DataFrameMapper to
    #normalize = [([col], RobustScaler()) for col in cols_normalize]#Uses sklearn_pandas.DataFrameMapper to
    normalize = [([col], MinMaxScaler()) for col in cols_normalize]#Uses sklearn_pandas.DataFrameMapper to
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(normalize + leave)
    
    #Transform features and convert to float32:
    X_train = x_mapper.fit_transform(data_train_temp).astype('float32')
    X_val = x_mapper.fit_transform(data_val_temp).astype('float32')
    X_test = x_mapper.fit_transform(data_test_temp).astype('float32')
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def get_feature_importance_sksurv(X_test, Y_test, model):
    """
    Method to estimate permutation feature importance for a given model on the test set. Works for scikit-survival models only.

    Parameters
    ----------
    X_test: pd.DATAFRAME
      Test set features (columns:features, rows entries)
    Y_test: pd.DATAFRAME
      Test set targets 
    model: sksurv MODEL 
      Trained scikit-survival model
      
    Returns
    -------
    importances: LIST 
      List of tuples containing the feature name, the average permutation importance across iterations & its standard deviation
    """
    perm = PermutationImportance(model, n_iter=15)
    perm.fit(X_test, Y_test)
    
    importances = list(zip(X_test.columns.tolist(), perm.feature_importances_, perm.feature_importances_std_))
    
    return importances


def get_feature_importance_pycox(data_train, data_test, model):
    """
    Method to estimate permutation feature importance for a given model on the test set. Works for scikit-survival models only.

    Parameters
    ----------
    data_test: pd.DATAFRAME
      Test dataframe preprocessed, filtered, undergone feature selection & additional dimensionality reduction (pycox models need data in this format)
      Test set targets 
    model: pycox MODEL 
      Trained pycox survival model
      
    Returns
    -------
    importances: LIST 
      List of tuples containing the feature name, the average permutation importance across iterations & its standard deviation
    """

    #Get the variables' names
    data_test_temp = data_test.copy()
    data_test_temp = data_test_temp.drop(['OS.time', 'OS'], axis = 1) # drop time & event columns
    feature_names = data_test_temp.columns.tolist()
    
    #Define score function for pycox deep survival models
    def score_pycox(X_test, Y_test):
        from pycox.evaluation import EvalSurv
        from pycox.models import CoxPH
        #from sksurv.metrics import concordance_index_censored

#         predictions_te = 1 - model.predict_surv_df(X_test) # predict_surv_df returns survival estimates, not risks
        
#         #events_te, durations_te = zip(*Y_test) 
#         events_te = Y_test[1]
#         durations_te = Y_test[0]
#         test_set_c_index = concordance_index_censored(events_te, durations_te, predictions_te)
#         test_set_c_index = test_set_c_index[0]
        
        test_set_c_index = EvalSurv(model.predict_surv_df(X_test), Y_test[0], Y_test[1], censor_surv='km').concordance_td() # TODO: Check if pred / 1-pred
        
        return test_set_c_index

    data_val_dummy = data_train.sample(frac=0.15)
    data_train_dummy = data_train.drop(data_val_dummy.index)
    _, _, _, _, X_test_pycox, Y_test_pycox = prepare_data_pycox(data_train_dummy, data_val_dummy, data_test)
    base_score, score_decreases = get_score_importances(score_pycox, X_test_pycox, Y_test_pycox, n_iter=15)

    feature_importances_mean = np.mean(score_decreases, axis=0)
    feature_importances_std = np.std(score_decreases, axis=0)

    importances = list(zip(feature_names, feature_importances_mean, feature_importances_std))
    
    return(importances)


def define_hparam_space_pycox(subsample_n_architectures): #TODO: add all hyperparameter choices as arguments 
    """
    Method to generate a subsample of hyperparameters to perform the grid search on.
    It generates all possible network architectures. These will consist of 1-5 hidden layers, each with number
    of nodes equal to one of [32, 64, 128, 256, 512, 1024]. We further impose the constraint that subsequent layers
    have <= nodes than previous ones. The other hyperparameters have a fixed space 'dropout': [0.1, 0.2, 0.3, 0.4],
    'batch_size': [8, 16, 32], 'learning_rate': [0.1, 0.01, 0.001].

    Parameters
    ----------
    subsample_n_architectures: INT
      Number of architectures to sample
      
    Returns
    -------
    hyperparam_space: DICTIONARY 
      Grid for hyperparameter optimization; includes hyperparameters: 'num_nodes', 'dropout', 'batch_size', 'learning_rate'
    """
    import itertools
    import random

    #Generate all architectures consisting of 1-5 hidden layers, each with number of nodes equal to one of [32, 64, 128, 256, 512, 1024]
    architectures = [] 
    iterables = [ [32, 64, 128, 256, 512, 1024], [0, 32, 64, 128, 256, 512, 1024], [0, 32, 64, 128, 256, 512, 1024], [0, 32, 64, 128, 256, 512, 1024], [0, 32, 64, 128, 256, 512, 1024]]
    for architecture in itertools.product(*iterables):
        architecture = list(filter(lambda a: a != 0, architecture))#Remove layers with zero nodes
        architecture = list(set(architecture)) #Keep unique architectures
        
        #Keep only architectures that have subsequent layers have <= nodes than previous ones: 
        n_max = architecture[0]
        flag_to_drop = 0
        for i in range(len(architecture)):
            if architecture[i] > n_max:
                flag_to_drop = 1
            else: 
                n_max = architecture[i]
        if flag_to_drop == 0:   
            architectures.append(architecture)
    #Keep a random subsample of the generated architectures:            
    architectures = random.sample(architectures, subsample_n_architectures)  
    
    #Generate grid for hyperparameter optimization:       
    hyperparam_space = {'num_nodes': architectures,
                        'dropout': [0.1, 0.2, 0.3, 0.4],
                        'batch_size': [8, 16, 32],
                        'learning_rate': [0.1, 0.01, 0.001]}
    return hyperparam_space

#As above but architecture depends on input size. TODO: Replace function above with this one
def define_hparam_space_pycox_dynamic(subsample_n_architectures, input_layer_size):
    """
    Method to generate a subsample of hyperparameters to perform the grid search on.
    It generates all possible network architectures. These will consist of 1-5 hidden layers, each with number
    of nodes equal to one of [16, 32, 64, 128, ... input_layer_size/2 rounded down to closest power of 2]. We further
    impose the constraint that subsequent layers have <= nodes than previous ones. The other hyperparameters have
    a fixed space 'dropout': [0.1, 0.2, 0.3, 0.4], 'batch_size': [8, 16, 32], 'learning_rate': [0.1, 0.01, 0.001].

    Parameters
    ----------
    subsample_n_architectures: INT
      Number of architectures to sample
      
    Returns
    -------
    hyperparam_space: DICTIONARY 
      Grid for hyperparameter optimization; includes hyperparameters: 'num_nodes', 'dropout', 'batch_size', 'learning_rate'
    """
    import itertools
    import random
    import numpy as np

    #Get maximal number of nodes (input_layer_size/2 rounded down to closest power of 2):
    max_num_nodes = int(2**np.floor(np.log2(input_layer_size/2)))
    
    num_nodes_set = []
    n = max_num_nodes
    
    if n <= 16:
        num_nodes_set = [16]
    else:
        while n > 16:
            num_nodes_set.append(n)
            n = int(np.min(np.asarray(num_nodes_set))/2)
        
    num_nodes_set_plus_0 = num_nodes_set + [0]
    
    #Generate all architectures consisting of 1-5 hidden layers, each with number of nodes equal to one of [16, 32, 64, 128, 256, 512, 1024]
    architectures = [] 
    iterables = [num_nodes_set, num_nodes_set_plus_0, num_nodes_set_plus_0, num_nodes_set_plus_0, num_nodes_set_plus_0]
    for architecture in itertools.product(*iterables):
        architecture = list(filter(lambda a: a != 0, architecture))#Remove layers with zero nodes
        
        #Keep only architectures that have subsequent layers have <= nodes than previous ones: 
        n_max = architecture[0]
        flag_to_drop = 0
        for i in range(len(architecture)):
            if architecture[i] > n_max:
                flag_to_drop = 1
            else: 
                n_max = architecture[i]
        if flag_to_drop == 0:   
            architectures.append(architecture)
    #Keep a random subsample of the generated architectures:            
    architectures = random.sample(architectures, subsample_n_architectures)  
    
    #Keep only unique architectures
    unique_architectures = []

    for item in architectures:
        if item not in unique_architectures:
            unique_architectures.append(item)
    
    #Generate grid for hyperparameter optimization:       
    hyperparam_space = {'num_nodes': unique_architectures,
                        'dropout': [0.1, 0.2, 0.3, 0.4],
                        'batch_size': [8, 16, 32],
                        'learning_rate': [0.1, 0.01, 0.001]}
    return hyperparam_space



# def get_feature_importance_AE_sksurv(X_test_orig, Y_test, encoder_model, survival_model):
#     """
#     Method to estimate permutation feature importance for a given survival model, chained after a given autoencoder on the test set. Works for scikit-survival models only.

#     Parameters
#     ----------
#     X_test_orig: pd.DATAFRAME
#       Test data preprocessed, filtered, undergone feature selection but before additional dimensionality reduction (sksurv models need data in this format)
#     Y_test: TUPLE
#       Test set targets (event & column tuple) (sksurv models need data in this format)
#     encoder_model: OBJECT
#       Encoder model
#     model: sksurv MODEL 
#       Trained sksurv survival model
      
#     Returns
#     -------
#     importances: LIST 
#       List of tuples containing the feature name, the average permutation importance across iterations & its standard deviation
#     """
#     X_test = X_test_orig.copy()
    
#     #Define score function for pycox deep survival models
#     def score_AE_sksurv(X_test, Y_test):
#         import pandas as pd
#         from sksurv.metrics import concordance_index_censored
        
#         X_test_df = pd.DataFrame(data=X_test, columns=feature_list)
        
#         X_test = encode(X_test_df, encoder_model)
        
#        ### predictions_te = 1 - trained_model.predict_surv_df(X_test_pycox) # predict_surv_df returns survival estimates, not risks
        
# #         #Evaluate:
# #         predictions_te = survival_model.predict(X_test)
# #         events_te, durations_te = zip(*Y_test) 
# #         test_set_c_index = concordance_index_censored(events_te, durations_te, predictions_te)
# #         test_set_c_index = test_set_c_index[0]

#         test_set_c_index = survival_model.score(X_test, Y_test)
#         #print('In score_AE_sksurv: '+str(test_set_c_index))
#         return test_set_c_index

#     #print(X_test_orig)
#     #print(Y_test)
#     feature_list = list(X_test_orig.columns)
#     X_test_orig_array = X_test_orig.to_numpy()
    
#     base_score, score_decreases = get_score_importances(score_AE_sksurv, X_test_orig_array, Y_test, n_iter=15)#X_test_orig

#     feature_importances_mean = np.mean(score_decreases, axis=0)
#     feature_importances_std = np.std(score_decreases, axis=0)

#     importances = list(zip(feature_list, feature_importances_mean, feature_importances_std))
    
#     return(importances)