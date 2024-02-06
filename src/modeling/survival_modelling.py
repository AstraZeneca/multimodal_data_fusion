import os, sys
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from os.path import dirname, abspath, join
parent_of_filedir = dirname(dirname(abspath(__file__)))
sys.path.append(join(parent_of_filedir, 'utils'))
from survival_utilities import get_feature_importance_pycox, get_feature_importance_sksurv, prepare_data_sksurv, prepare_data_pycox, \
                               define_hparam_space_pycox_dynamic, define_hparam_space_pycox, hyperparameter_search 

def survival_modelling_main(data_train, data_val, data_test, target_cols, model, get_importances, individual_predictions, verbose):
    """
    Method to train & evaluate a specified survival model on the data. 

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Training dataframe preprocessed, filtered, undergone feature selection & additional dimensionality reduction (pycox models need data in this format)
    data_val: pd.DATAFRAME
      Validation dataframe preprocessed, filtered, undergone feature selection & additional dimensionality reduction (pycox models need data in this format)
    data_test: pd.DATAFRAME
      Test dataframe preprocessed, filtered, undergone feature selection & additional dimensionality reduction (pycox models need data in this format)
    model: STRING
      Model to train. Options include:
      'CPH-L2' for Cox Proportional Hazards survival model with Ridge Penalty
      'CPH-EN' for Cox Proportional Hazards survival model with elastic net penalty 
      'CPH-GB' for Gradient Boosted unregularized Cox Proportional Hazards survival model
      'CLS-GB' for Gradient Boosted Componentwise Least Squares survival model
      'RSF' for Randomized Survival Forest survival model
      'DEEP-SURV' for continuous time CoxPH / Deep Survival model [J. L. Katzman,et al (2018), https://arxiv.org/abs/1606.00931] - note this needs a different data input format, hyperparameter optimization, etc.
    get_importances: INT
       1 to get feature importance list of the model, 0 not to do so
    verbose: INT
       1 to print execution details, 0 not to do so
       
    Returns
    -------
    surv_model: OBJECT (sksurv or pycox MODEL)
    The trained model on the training dataset under the best hyperparameter configuration identified.
    test_set_c_index: FLOAT
    The Concordance Index attained by the model on the test set.
    val_set_c_index: FLOAT
    The Concordance Index attained by the model on the validation set.
    training_set_c_index: FLOAT
    The Concordance Index attained by the model on the training set.
    importances_df: pd.DATAFRAME
    Dataframe containing model information along with the model's feature importance list. Columns include 'Feature','Feature Importance', 'Feature Importance STD', 'Model', 'Test C-index', 'Validation C-index', 'Training C-index', 'Rank'.
    
    """
    pred_train, pred_test = [], []    
    if model in ['RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN']: #For sksurv models (shallow) 
        #Prepare data for sksurv model format
        X_train, Y_train, X_test, Y_test = prepare_data_sksurv(data_train, data_test, target_cols)
        X_train, Y_train, X_val, Y_val = prepare_data_sksurv(data_train, data_val, target_cols)
        #Train survival model
        surv_model, train_set_c_index, val_set_c_index = train_survival_model_sksurv(X_train, Y_train, model) 
        #Evaluate survival model on the validation set:
        val_set_c_index = surv_model.score(X_val, Y_val)
        #Evaluate survival model on the test set:
        test_set_c_index = surv_model.score(X_test, Y_test)
        print("Test set C-index: "+str(round(test_set_c_index, 3)))
                
    elif model in ['DEEP-SURV']:#For pycox models (deep) -could add more here in future
        #Train & evaluate survival model
        surv_model, train_set_c_index, val_set_c_index, test_set_c_index = train_survival_model_pycox(data_train, data_test, model, verbose) 
    else:
        print("Model: "+model+" is not recognised! Currently only 'RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN' & 'DEEP-SURV' are supported.")
        
    #Calculate feature importances (permutation-based feature importances) - Currently only supporting sksurv models
    #First create an empty dataframe:
    importances_df = pd.DataFrame(columns=['Feature', 'Feature Importance', 'Feature Importance STD', 'Model', 'Test C-index', 'Validation C-index', 'Train C-index', 'Rank'])
    if get_importances == 1: #If user asked for feature importance list, populate the dataframe 
        if model in ['RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN']: #For sksurv models (shallow)
            print("Calculating feature importances for "+model+" model (sksurv).")
            #show_f_importance(X_test, Y_test, model)#This function just prints top 25 - does not print in loop; replace with below
            importances = get_feature_importance_sksurv(X_test, Y_test, surv_model)
            importances_df = pd.DataFrame(importances, columns=['Feature', 'Feature Importance', 'Feature Importance STD']) #Add importances to a dataframe
            importances_df['Model'] = model #Specify which model produced the results
            importances_df['Test C-index'] = test_set_c_index #Specify test set C-index attained by model
            importances_df['Validation C-index'] = val_set_c_index #Specify validation set C-index attained by model of same hyperparameters
            importances_df['Train C-index'] = train_set_c_index #Specify training set C-index attained by model
            importances_df['Rank'] = importances_df['Feature Importance'].rank(ascending=False)
            #Print importances:
            if verbose == 1:
                print(importances_df)
        else: #For pycox models (deep)
            print("Calculating feature importances for "+model+" model (pycox).")
            #show_f_importance(X_test, Y_test, model)#This function just prints top 25 - does not print in loop; replace with below
            importances = get_feature_importance_pycox(data_train, data_test, surv_model)
            importances_df = pd.DataFrame(importances, columns=['Feature', 'Feature Importance', 'Feature Importance STD']) #Add importances to a dataframe
            importances_df['Model'] = model #Specify which model produced the results
            importances_df['Test C-index'] = test_set_c_index #Specify test set C-index attained by model
            importances_df['Validation C-index'] = val_set_c_index #Specify validation set C-index attained by model of same hyperparameters
            importances_df['Train C-index'] = train_set_c_index #Specify training set C-index attained by model
            importances_df['Rank'] = importances_df['Feature Importance'].rank(ascending=False)
            #Print importances:
            if verbose == 1:
                print(importances_df)
    else: #If no feature importances are requested by the user:
        print('No feature importances requested. Generating an empty dataframe.')

        #TODO: In future versions, replace c-indices with predictions on individual datapoints - calculate c-indices outside
    if individual_predictions:
        pred_train = surv_model.predict(X_train)
        pred_val = surv_model.predict(X_val)
        pred_test = surv_model.predict(X_test)  
    return(surv_model, test_set_c_index, val_set_c_index, train_set_c_index, importances_df, pred_train, pred_val, pred_test)


def train_survival_model_sksurv(X_train, Y_train, model_class):   
    """
    Method to perform CV model selection / hyperparameter optimization of scikit-survival models and train the final
    model based on best hyperparameters identified.

    Parameters
    ----------
    X_train: pd.DATAFRAME
      Training set features (columns:features, rows entries)
    Y_train: pd.DATAFRAME
      Training set targets; format (columns: duration & event, rows entries)  
    model_class: STRING
      Specify model class from which the model will be drawn. Currently only the following options are supported. 
      'CPH-L2' for Cox Proportional Hazards survival model with Ridge Penalty
      'CPH-EN' for Cox Proportional Hazards survival model with elastic net penalty 
      'CPH-GB' for Gradient Boosted unregularized Cox Proportional Hazards survival model
      'CLS-GB' for Gradient Boosted Componentwise Least Squares survival model
      'RSF' for Randomized Survival Forest survival model

    Returns
    -------
    model: sksurv MODEL
      Final trained scikit survival model 
    train_set_c_index: FLOAT
      Concordance index of the model calculated on the training set
    val_set_c_index: FLOAT
      Concordance index of the model calculated on the validation set
    """
    import pandas as pd
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import RepeatedKFold # RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV # RandomizedSearchCV
    #from sklearn.model_selection import cross_val_score
    #from skopt import BayesSearchCV #TODO: Can switch to Bayesian hyperparameter optimization when they make this compatible to latest sklearn supporting sksurv
    
    splits = 5 #number of CV splits
    
    n_features = X_train.shape[1]
    n_points_per_fold = int(X_train.shape[0]/splits) #number of datapoints per fold rounded to closest integer
    
    if model_class == 'CPH-L2':
        print("TRAINING SURVIVAL MODEL -- COX PROPORTIONAL HAZARDS WITH RIDGE REGULARIZATION")
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        import numpy as np
                
        print("Initiating hyperparameter optimization.")
        model = CoxPHSurvivalAnalysis()
        
        #Define search space 
        hyperparam_space = {'alpha': 10. ** np.linspace(-4, 4, 10)}#loguniform(1e-5, 100)
        # Find optimal parameters
        result = hyperparameter_search(data=[X_train, Y_train], hyperparam_space=hyperparam_space, model=model, splits=splits)
        print('Retraining model on full training set with optimal hyperparameters identified:')
        
        #Use model with best identified hyperparameter configuration:
        model = CoxPHSurvivalAnalysis(alpha=result.best_params_['alpha'])    

    elif model_class == 'CPH-EN':
        print("TRAINING SURVIVAL MODEL -- COX PROPORTIONAL HAZARDS WITH ELASTIC NET REGULARIZATION")
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        # from scipy.stats import loguniform
        
        print("Initiating hyperparameter optimization.")
        model = CoxnetSurvivalAnalysis()
        
        #Define search space 
        hyperparam_space = {'l1_ratio' : [1e-16, 0.25, 0.5, 0.75, 1.0],
                            'alpha_min_ratio': [0.0001, 0.001, 0.01, 0.1]}
        
        # Find optimal parameters
        result = hyperparameter_search(data=[X_train, Y_train], hyperparam_space=hyperparam_space, model=model, splits=splits)
        print('Retraining model on full training set with optimal hyperparameters identified:')
        
        #Use model with best identified hyperparameter configuration:
        model = CoxnetSurvivalAnalysis(l1_ratio=result.best_params_['l1_ratio'],
                                       alpha_min_ratio=result.best_params_['alpha_min_ratio'])        
       
    elif model_class == 'CPH-GB':
        print("TRAINING SURVIVAL MODEL -- GRADIENT BOOSTED UNREGULARIZED COX PROPORTIONAL HAZARDS")
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        print("Initiating hyperparameter optimization.")
        model = GradientBoostingSurvivalAnalysis()
        
        #Define search space 
        hyperparam_space = {'n_estimators': [10, 50, 100],
                            'learning_rate': [0.8, 0.9, 1.0],
                            'max_depth': (1, int(n_features/2))}
        
        # Find optimal parameters
        result = hyperparameter_search(data=[X_train, Y_train], hyperparam_space=hyperparam_space, model=model, splits=splits)
        print('Retraining model on full training set with optimal hyperparameters identified:')
        
        #Use model with best identified hyperparameter configuration:
        model = GradientBoostingSurvivalAnalysis(n_estimators=result.best_params_['n_estimators'],
                                                 learning_rate=result.best_params_['learning_rate'],
                                                 max_depth=result.best_params_['max_depth'])

    elif model_class == 'CLS-GB':
        print("TRAINING SURVIVAL MODEL -- GRADIENT BOOSTED COMPONENTWISE LEAST SQUARES")
        from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

        print("Initiating hyperparameter optimization.")
        model = ComponentwiseGradientBoostingSurvivalAnalysis()
        
        #Define search space 
        hyperparam_space = {'n_estimators': [10, 50, 100],
                            'learning_rate': [0.8, 0.9, 1.0]}
        
        # Find optimal parameters
        result = hyperparameter_search(data=[X_train, Y_train], hyperparam_space=hyperparam_space, model=model, splits=splits)
        print('Retraining model on full training set with optimal hyperparameters identified:')
        
        #Use model with best identified hyperparameter configuration:
        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=result.best_params_['n_estimators'],
                                                              learning_rate=result.best_params_['learning_rate'])
    
    elif model_class == 'RSF':
        print("TRAINING SURVIVAL MODEL -- RANDOM SURVIVAL FOREST")
        from sksurv.ensemble import RandomSurvivalForest

        print("Initiating hyperparameter optimization.")
        model = RandomSurvivalForest()
        
        #Define search space 
        hyperparam_space = {'n_estimators': [10, 50, 100],
                            'min_samples_split': [10, 20, 50, 100],
                            'min_samples_leaf': [5, 10, 20],
                            'max_features': ['sqrt']}
    
        # Find optimal parameters
        result = hyperparameter_search(data=[X_train, Y_train], hyperparam_space=hyperparam_space, model=model, splits=splits)
        print('Retraining model on full training set with optimal hyperparameters identified:')
        
        #Use model with best identified hyperparameter configuration:
        model = RandomSurvivalForest(n_estimators=result.best_params_['n_estimators'],
                                     min_samples_split=result.best_params_['min_samples_split'],
                                     min_samples_leaf=result.best_params_['min_samples_leaf'],
                                     max_features=result.best_params_['max_features'])   
    else:
        raise ValueError("Unrecognised user choice for model_class; no model trained.")
    
    #Save model's performance on validation set:
    val_set_c_index = result.best_score_
        
    #Now fit the model on full training set:
    model.fit(X_train, Y_train)
    
    train_set_c_index = model.score(X_train, Y_train)
    print("MODEL TRAINING FINISHED!")
    print("Training set C-index: "+str(round(train_set_c_index, 3)))
    
    #TODO: Return predictions rather than C-indices, evaluate latter outside.
    
    return(model, train_set_c_index, val_set_c_index)


def train_survival_model_pycox(data_train, data_test, model_class, verbose):   
    """
    Main method to prepare data for pycox survival models, perform CV model selection for hyperparameter optimization and
    train the final based on best hyperparameters identified on the training set and evaluate it on the test set.
    
    Parameters
    ----------
    data_train: pd.DATAFRAME
      Training set (contains both feature & target columns)
    data_test: pd.DATAFRAME
      Test set (contains both feature & target columns)
    model_class: STRING
      Specify model class from which the model will be drawn. Currently only the following option is supported:
      'DEEP-SURV' for continuous time CoxPH / Deep Survival model [J. L. Katzman,et al (2018),  https://arxiv.org/abs/1606.00931].
    verbose: INT
      Flag; 1 to print detailed output, 0 to not do so
    
    Returns
    -------
    model: pycox MODEL 
      Return the trained pycox survival model
    train_set_c_index: FLOAT
      Concordance index of the model calculated on the training set
    val_set_c_index: FLOAT
      Concordance index of the model calculated on the validation set
    test_set_c_index: FLOAT
      Concordance index of the model calculated on the test set
    """
    import pandas as pd
    import numpy as np
    from random import shuffle
    from sklearn.model_selection import KFold
    from pycox.evaluation import EvalSurv

    #Setting up CV-folds
    splits = 3 #number of CV splits
    kf = KFold(n_splits = splits, shuffle = True)

    #Define hyperparameter search space:
    #hyperparameter_space = define_hparam_space_pycox(4)#Argument is number of randomly drawn architectures to retain
    input_layer_size = data_train.shape[1]-2
    hyperparameter_space = define_hparam_space_pycox_dynamic(4, input_layer_size)#1st Argument is number of randomly drawn architectures to retain, 2nd Argument is size of input layer
    
    results_full = pd.DataFrame(columns=['hyperparam_config_id', 'fold', 'num_nodes', 'dropout', 'batch_size', 'learning_rate', 'epochs', 'C_index_train', 'C_index_val'])

    if verbose == 1:
        print('Original training set size: '+str(data_train.shape))

    #Repeat for each fold
    print('Starting Cross-validation for hyperparameter tuning for model '+model_class+'.')
    for fold in range(splits):
        folds = next(kf.split(data_train), None)

        data_train_fold = data_train.iloc[folds[0]]
        data_val_fold = data_train.iloc[folds[1]]

        if verbose == 1:
            print('CV fold '+str(fold+1)+', training set size: '+str(data_train_fold.shape))
            print('CV fold '+str(fold+1)+', validation set size: '+str(data_val_fold.shape))

        hyperparam_config_id = 0
        #Repeat for each hyperparameter configuration
        for indx_architectures in range(len(hyperparameter_space['num_nodes'])):
            for indx_dropout in range(len(hyperparameter_space['dropout'])):
                for indx_batch_size in range(len(hyperparameter_space['batch_size'])):
                    for indx_learning_rate in range(len(hyperparameter_space['learning_rate'])):

                        hyperparam_config = {'num_nodes': hyperparameter_space['num_nodes'][indx_architectures],
                                            'dropout': hyperparameter_space['dropout'][indx_dropout],
                                            'batch_size': hyperparameter_space['batch_size'][indx_batch_size],
                                            'learning_rate': hyperparameter_space['learning_rate'][indx_batch_size]}

                        hyperparam_config_id = hyperparam_config_id + 1
                        print('CV fold = '+str(fold+1)+'/'+str(splits)+', hyperparam_config = '+str(hyperparam_config_id)+'/'+str(len(hyperparameter_space['num_nodes'])*len(hyperparameter_space['dropout'])*len(hyperparameter_space['batch_size'])*len(hyperparameter_space['learning_rate'])))

                        if verbose == 1:
                            print('Hyperparameter configuration under examination:')
                            print(hyperparam_config)

                        #Prepare data for deep survival models
                        X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, X_test, Y_test = prepare_data_pycox(data_train_fold, data_val_fold, data_test)

                        #Train survival model
                        results = train_survival_model_pycox_CV(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, model_class, hyperparam_config, verbose)

                        #Add hyperparam_config_id & fold indication to results dataframe:
                        results['hyperparam_config_id'] = hyperparam_config_id
                        results['fold'] = fold

                        #Add results for latest fold / hyperparameter configuration to results_all dataframe:
                        results_full = pd.concat([results_full, results])

    if verbose == 1:
        print('Cross validation detailed results:')
        print(results_full)

    #Average across folds & identify best hyperparameter configuration
    best_hparam_config_id = results_full.groupby('hyperparam_config_id')['C_index_val'].mean().idxmax()
    
    #Get best hyperparameters
    best_hparams = {'num_nodes': results_full.iloc[best_hparam_config_id-1]['num_nodes'],
                    'dropout': results_full.iloc[best_hparam_config_id-1]['dropout'],
                    'batch_size': results_full.iloc[best_hparam_config_id-1]['batch_size'],
                    'learning_rate': results_full.iloc[best_hparam_config_id-1]['learning_rate'],
                    'epochs': results_full.iloc[best_hparam_config_id-1]['epochs']}

    if verbose == 1:
        print('Best hyperparameter configuration identified:')
        print(best_hparams)
        
    #Train a final model on a larger training set (only 15% of original training data used for validation):

    data_val_final = data_train.sample(frac=0.15)
    data_train_final = data_train.drop(data_val_final.index)

    #Prepare data for deep survival models
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data_pycox(data_train_final, data_val_final, data_test)
            
    #Train survival model -Difference from above: use full training set for training, no early stopping (train for best number of epochs identified)
    model, train_set_c_index, val_set_c_index = train_survival_model_pycox_final(X_train, Y_train, X_val, Y_val, model_class, best_hparams)
    
    #Evaluate survival model
    test_set_c_index = EvalSurv(model.predict_surv_df(X_test), Y_test[0], Y_test[1], censor_surv='km').concordance_td()
    _ = model.compute_baseline_hazards()
    print("Test set C-index: "+str(round(test_set_c_index, 3)))
    
    #TODO: Return predictions rather than C-indices, evaluate latter outside.
    
    return(model, train_set_c_index, val_set_c_index, test_set_c_index)


def train_survival_model_pycox_CV(X_train, Y_train, X_val, Y_val, model_class, hyperparam_config, verbose_):   
    """
    Method to train CV pycox model for the purposes of model selection / hyperparameter optimization.
    It trains a model of the specified model class using the specified hyperparameters on the training data and evaluates it
    & performs early stopping on the validation data.
    
    Parameters
    ----------
    X_train: pd.DATAFRAME
      Test set features (columns:features, rows entries)
    Y_train: TUPLE
      Test set targets; format (duration, event)
    X_val: pd.DATAFRAME
      Validation set features (columns:features, rows entries)
    Y_val: TUPLE
      Validation set targets; format (duration, event)
    model_class: STRING
      Specify model class from which the model will be drawn. Currently only the following option is supported:
      'DEEP-SURV' for continuous time CoxPH / Deep Survival model [J. L. Katzman,et al (2018),  https://arxiv.org/abs/1606.00931].
    hyperparam_config: DICTIONARY
      Dictionary containing the hyperparameter configuration to be used. Keys include:
      'num_nodes', 'dropout', 'batch_size', 'learning_rate'
    verbose_: INT
        1 to print out detailed output, 0 not to
    Returns
    -------
    results: pd.DATAFRAME 
      Dataframe containing information on the model's hyperparameters & performance.
      Columns include 'num_nodes', 'dropout', 'batch_size', 'learning_rate', 'epochs', 'C_index_train', 'C_index_val'
    """
    import pandas as pd
    import numpy as np
    import torch # For building the networks 
    import torchtuples as tt # Some useful functions
    from pycox.evaluation import EvalSurv

    #Fix the following:
    in_features = X_train.shape[1]
    out_features = 1
    
    batch_norm = True
    output_bias = False
    epochs = 512
    es_patience = 5
    callbacks = [tt.callbacks.EarlyStopping(patience = es_patience)] 
    if verbose_ == 1:
        verbose = True
    else:
        verbose = False
    
    val = X_val, Y_val #pack validation features & targets for easier passing as arguments to model.fit
    
    if model_class == 'DEEP-SURV':
        from pycox.models import CoxPH

        #Define model & optimizer -- Use optimal hyperparameters identified by CV
        num_nodes = hyperparam_config['num_nodes'] #[256, 128, 64, 32] 
        dropout = hyperparam_config['dropout'] #0.2 
        batch_size = hyperparam_config['batch_size'] #32 
        learning_rate = hyperparam_config['learning_rate']

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

        model = CoxPH(net, tt.optim.Adam)
        
        #Use best learning rate to train model -- do not pick the learning rate that achieves the lowest loss, but instead something in the middle of the sharpest downward slope
        model.optimizer.set_lr(learning_rate)
        
        #Train model & store history
        log = model.fit(X_train, Y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val, val_batch_size=batch_size)
    #################################################################################################
    else:
        print("Warning! Unrecognised user choice for model_class; no model trained.")
    
    _ = model.compute_baseline_hazards()
    
    C_index_train = EvalSurv(model.predict_surv_df(X_train), Y_train[0], Y_train[1], censor_surv='km').concordance_td()
    if verbose_ == 1:
        print("Training set C-index: "+str(round(C_index_train ,3)))
    C_index_val = EvalSurv(model.predict_surv_df(X_val), Y_val[0], Y_val[1], censor_surv='km').concordance_td()
    if verbose_ == 1:
        print("Validation set C-index: "+str(round(C_index_val ,3)))    
    
    #Generate results dataframe:
    results = pd.DataFrame(columns=['hyperparam_config_id', 'fold', 'num_nodes', 'dropout', 'batch_size', 'learning_rate', 'epochs', 'C_index_train', 'C_index_val'])
    results['num_nodes'] = results['num_nodes'].astype(object)#Convert 'num_nodes' key to object as it will store a list; the rest will be floats
    results['num_nodes'] = [num_nodes]
    results['dropout'] = dropout
    results['batch_size'] = batch_size
    results['learning_rate'] = learning_rate 
    results['epochs'] = log.epochs[-1] # Get number of epochs ran; Best epoch was log.epochs[-1] - es_patience, but leave this for some slack
    results['C_index_train'] = C_index_train
    results['C_index_val'] = C_index_val

    return(results)


def train_survival_model_pycox_final(X_train, Y_train, X_val, Y_val, model_class, best_hparams):
    """
    Method to train the final pycox survival model based on best hyperparameters identified.
    It trains a model of the specified model class using the specified hyperparameters on the training data and evaluates it
    & performs early stopping on the validation data.
    
    Parameters
    ----------
    X_train: pd.DATAFRAME
      Training set features (columns:features, rows entries)
    Y_train: TUPLE
      Test set targets; format (duration, event)
    X_val: pd.DATAFRAME
      Validation set features (columns:features, rows entries)
    Y_val: TUPLE
      Validation set targets; format (duration, event)
    model_class: STRING
      Specify model class from which the model will be drawn. Currently only the following option is supported:
      'DEEP-SURV' for continuous time CoxPH / Deep Survival model [J. L. Katzman,et al (2018),  https://arxiv.org/abs/1606.00931].
    best_hparams: DICTIONARY
      Dictionary containing the hyperparameters that produced the best mean performance during CV. Keys include:
      'num_nodes', 'dropout', 'batch_size', 'learning_rate', 'epochs'. For now epochs are ignored and optimal
      value chosen based on early stopping
    
    Returns
    -------
    model: pycox MODEL 
      Final trained pycox survival model
    train_set_c_index: FLOAT
      Concordance index of the model calculated on the training set
    val_set_c_index: FLOAT
      Concordance index of the model calculated on the validation set
    """
    import pandas as pd
    import numpy as np
    import torch # For building the networks 
    import torchtuples as tt # Some useful functions
    from pycox.evaluation import EvalSurv

    #Fix the following:
    in_features = X_train.shape[1]
    out_features = 1
    
    batch_norm = True
    output_bias = False
    epochs = 512
    es_patience = 5
    callbacks = [tt.callbacks.EarlyStopping(patience = es_patience)] 
    verbose = True
    
    val = X_val, Y_val #treat training set as validation set for running fit; this has no effect on training as no callbacks are applied.
    
    if model_class == 'DEEP-SURV':
        from pycox.models import CoxPH
        print('Retraining DEEP-SURV model on full training set with optimal hyperparameters identified...')
        
        #Define model & optimizer -- Use optimal hyperparameters identified by CV
        num_nodes = best_hparams['num_nodes'] #[256, 128, 64, 32] 
        dropout = best_hparams['dropout'] #0.2 
        batch_size = best_hparams['batch_size'] #32 
        #epochs = best_hparams['epochs']
        learning_rate = best_hparams['learning_rate'] #Comment and find best based on training set.
        
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

        model = CoxPH(net, tt.optim.Adam)
        
        #Use best learning rate to train model
        model.optimizer.set_lr(learning_rate)
        
        #Train model & store history
        log = model.fit(X_train, Y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val, val_batch_size=batch_size)
    #################################################################################################
    else:
        print("Warning! Unrecognised user choice for model_class; no model trained.")
    
    _ = model.compute_baseline_hazards()
    
    train_set_c_index = EvalSurv(model.predict_surv_df(X_train), Y_train[0], Y_train[1], censor_surv='km').concordance_td()
    print("Final Model -- Training set C-index: "+str(round(train_set_c_index ,3)))
    val_set_c_index = EvalSurv(model.predict_surv_df(X_val), Y_val[0], Y_val[1], censor_surv='km').concordance_td()
    print("Validation set C-index: "+str(round(val_set_c_index ,3)))   
    
    return(model, train_set_c_index, val_set_c_index)


def get_weighted_ensemble_predictions(data_train, data_test, target_cols, survival_models, trained_models, training_c_indices, verbose):
    
    #TODO: Add Function description
    
    import numpy as np
    from sksurv.metrics import concordance_index_censored
    
    print("Training and evaluating a weighted ensemble of {"+','.join(survival_models)+"}.")
    predictions = pd.DataFrame(np.nan, index=range(data_test.shape[0]), columns=survival_models)
    
    #Weights based on the training set (removing models with C-index < 0.5):
    ensemble_weights_tr = training_c_indices - 0.5*np.ones((training_c_indices.shape))
    ensemble_weights_tr[ensemble_weights_tr<0] = 0 #models with C-index < 0.5 are assigned a weight of 0 
    ensemble_weights_tr = ensemble_weights_tr/np.sum(ensemble_weights_tr)

    #Weights based on the training set (including models with C-index < 0.5, flipping its predictions):
    ensemble_weights_tr_all_models = np.absolute(training_c_indices - 0.5*np.ones((training_c_indices.shape)))
    ensemble_weights_tr_all_models = ensemble_weights_tr_all_models/np.sum(ensemble_weights_tr_all_models)
    
    if verbose == 1:
        print("The models' training set C-indices are:")
        print(training_c_indices)
        print("The models' weights in the ensemble based on the training set, excluding models with C-index < 0.5, are:")
        print(ensemble_weights_tr)
        print("The models' weights in the ensemble based on the training set, including models with C-index < 0.5 with flipped predictions, are:")
        print(ensemble_weights_tr_all_models)
    
    #Get predictions from each model on test set examples:
    for model in survival_models:
        if verbose == 1:
            print("Getting predictions of "+model+" model on test set.")
            
        _, _, X_test_sksurv, Y_test_sksurv = prepare_data_sksurv(data_train, data_test, target_cols) #Will need Y_test_sksurv in final evaluation even for pycox models
            
        if model in ['RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN']: #For sksurv models (shallow)
            from sksurv.linear_model import CoxPHSurvivalAnalysis
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            from sksurv.ensemble import GradientBoostingSurvivalAnalysis
            from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
            from sksurv.ensemble import RandomSurvivalForest
            
#             _, _, X_test_sksurv, Y_test_sksurv = prepare_data_sksurv(data_train, data_test) 
            predictions[model] = trained_models[model].predict(X_test_sksurv)

        else: #For pycox models (deep)
            from pycox.evaluation import EvalSurv
            
            data_val_dummy = data_train.sample(frac=0.15)
            data_train_dummy = data_train.drop(data_val_dummy.index)
            
            _, _, _, _, X_test_pycox, Y_test_pycox = prepare_data_pycox(data_train_dummy, data_val_dummy, data_test)
            predictions[model] = 1 - trained_models[model].predict_surv_df(X_test_pycox) # predict_surv_df returns survival estimates, not risks
    
    #First normalize predictions (risk scores) from each model to [0,1] - might refine this
    predictions_normalised = (predictions-predictions.min())/(predictions.max()-predictions.min())

    #Flip predictions of models with C-index < 0.5
    for model in range(len(survival_models)):
        if training_c_indices[model] < 0.5 :
            if verbose == 1:
                print("Model "+survival_models[model]+" achieved a training set C-index = "+str(training_c_indices[model])+", i.e. < 0.5. Its predictions will be flipped in the full version of the ensemble.")
            predictions_normalised[survival_models[model]] = 1 - predictions_normalised[survival_models[model]]
    
    #Get final (weighted) prediction:
    #print(predictions_normalised)
    train_all_weighted_final_prediction = np.multiply(ensemble_weights_tr_all_models, predictions_normalised).sum(axis=1)
    train_weighted_final_prediction = np.multiply(ensemble_weights_tr, predictions_normalised).sum(axis=1)

    #Evaluate final (weighted) prediction:
    events, durations = zip(*Y_test_sksurv) 
    train_all_weighted_ensemble_c_index = concordance_index_censored(events, durations, train_all_weighted_final_prediction)
    train_weighted_ensemble_c_index = concordance_index_censored(events, durations, train_weighted_final_prediction)

    return(train_all_weighted_ensemble_c_index[0], train_weighted_ensemble_c_index[0])


# def survival_modelling_main_AE_importances(data_train_orig, data_test_orig, encoder_model, model, get_importances, verbose):# As 'survival_modelling_main' but allows for input data to all be encoded under a joint AE -- TODO: Finish, Refine to allow for per_modality and/or excluded modalities, Test, then Replace 'survival_modelling_main'
#     """
#     Method to train & evaluate a specified survival model on the data. 

#     Parameters
#     ----------
#     data_train_orig: pd.DATAFRAME
#       Training dataframe preprocessed, filtered, undergone feature selection but before additional dimensionality reduction (pycox models need data in this format)
#     data_test_orig: pd.DATAFRAME
#       Test dataframe preprocessed, filtered, undergone feature selection but before additional dimensionality reduction (pycox models need data in this format)
#     encoder_model: OBJECT
#       'None' if data have not been transformed by AE, Trained Encoder model otherwise
#     model: STRING
#       Model to train. Options include:
#       'CPH-L2' for Cox Proportional Hazards survival model with Ridge Penalty
#       'CPH-EN' for Cox Proportional Hazards survival model with elastic net penalty 
#       'CPH-GB' for Gradient Boosted unregularized Cox Proportional Hazards survival model
#       'CLS-GB' for Gradient Boosted Componentwise Least Squares survival model
#       'RSF' for Randomized Survival Forest survival model
#       'DEEP-SURV' for continuous time CoxPH / Deep Survival model [J. L. Katzman,et al (2018), https://arxiv.org/abs/1606.00931] - note this needs a different data input format, hyperparameter optimization, etc.
#     get_importances: INT
#        1 to get feature importance list of the model, 0 not to do so
#     verbose: INT
#        1 to print execution details, 0 not to do so
       
#     Returns
#     -------
#     surv_model: OBJECT (sksurv or pycox MODEL)
#     The trained model on the training dataset under the best hyperparameter configuration identified.
#     test_set_c_index: FLOAT
#     The Concordance Index attained by the model on the test set.
#     val_set_c_index: FLOAT
#     The Concordance Index attained by the model on the validation set.
#     training_set_c_index: FLOAT
#     The Concordance Index attained by the model on the training set.
#     importances_df: pd.DATAFRAME
#     Dataframe containing model information along with the model's feature importance list. Columns include 'Feature','Feature Importance', 'Feature Importance STD', 'Model', 'Test C-index', 'Validation C-index', 'Training C-index', 'Rank'.
    
#     """
    
#     #TODO: Add additional dimensionality reduction in here, return also encoders
#     #TODO: Handle 'per-modality' encoders, handle 'excluded_modalities' in encoders for feature importance in original space
#     #TODO: Extensive testing
#     #TODO: Once all this is done, replace survival_modelling_main with this 
    
#     if model in ['RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN']: #For sksurv models (shallow)
#         #Prepare data for sksurv model format
#         X_train_orig, Y_train, X_test_orig, Y_test = prepare_data_sksurv(data_train_orig, data_test_orig)

#         # Work on copies to retain original data
#         X_train = X_train_orig.copy() 
#         X_test = X_test_orig.copy()
        
#         #Apply encoder tranformation to original data
#         X_train = encode(X_train, encoder_model)
#         X_test = encode(X_test, encoder_model)
        
#         #Train survival model
#         surv_model, train_set_c_index, val_set_c_index = train_survival_model_sksurv(X_train, Y_train, model) 

#         #Evaluate survival model on the test set:
#         test_set_c_index = surv_model.score(X_test, Y_test)
#         print("Test set C-index: "+str(round(test_set_c_index, 3)))
                
#     elif model in ['DEEP-SURV']:#For pycox models (deep) -could add more here in future
#         from pycox.models import CoxPH
        
#         #Train & evaluate survival model
#         surv_model, train_set_c_index, val_set_c_index, test_set_c_index = train_survival_model_pycox(data_train_orig, data_test_orig, model, verbose) 
#     else:
#         print("Model: "+model+" is not recognised! Currently only 'RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN' & 'DEEP-SURV' are supported.")
        
#     #Calculate feature importances (permutation-based feature importances) - Currently only supporting sksurv models
#     #First create an empty dataframe:
#     importances_df = pd.DataFrame(columns=['Feature', 'Feature Importance', 'Feature Importance STD', 'Model', 'Test C-index', 'Validation C-index', 'Train C-index', 'Rank'])
#     if get_importances == 1: #If user asked for feature importance list, populate the dataframe 
#         if model in ['RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN']: #For sksurv models (shallow)
#             print("Calculating feature importances for "+model+" model (sksurv).")
#             importances = get_feature_importance_AE_sksurv(X_test_orig, Y_test, encoder_model, surv_model)
#             #importances = get_feature_importance_sksurv(X_test, Y_test, surv_model)
#             importances_df = pd.DataFrame(importances, columns=['Feature', 'Feature Importance', 'Feature Importance STD']) #Add importances to a dataframe
#             importances_df['Model'] = model #Specify which model produced the results
#             importances_df['Test C-index'] = test_set_c_index #Specify test set C-index attained by model
#             importances_df['Validation C-index'] = val_set_c_index #Specify validation set C-index attained by model of same hyperparameters
#             importances_df['Train C-index'] = train_set_c_index #Specify training set C-index attained by model
#             importances_df['Rank'] = importances_df['Feature Importance'].rank(ascending=False)
#             #Print importances:
#             if verbose == 1:
#                 print(importances_df)             
#         else: #For pycox models (deep)
#             print("Calculating feature importances for "+model+" model (pycox).")
#             importances = get_feature_importance_pycox(data_train_orig, data_test_orig, surv_model)
#             importances_df = pd.DataFrame(importances, columns=['Feature', 'Feature Importance', 'Feature Importance STD']) #Add importances to a dataframe
#             importances_df['Model'] = model #Specify which model produced the results
#             importances_df['Test C-index'] = test_set_c_index #Specify test set C-index attained by model
#             importances_df['Validation C-index'] = val_set_c_index #Specify validation set C-index attained by model of same hyperparameters
#             importances_df['Train C-index'] = train_set_c_index #Specify training set C-index attained by model
#             importances_df['Rank'] = importances_df['Feature Importance'].rank(ascending=False)
#             #Print importances:
#             if verbose == 1:
#                 print(importances_df)
#     else: #If no feature importances are requested by the user:
#         print('No feature importances requested. Generating an empty dataframe.')
        
#     return(surv_model, test_set_c_index, val_set_c_index, train_set_c_index, importances_df)


# def survival_modelling_detailed(data_train, data_test, model, get_importances, verbose):
#   """
#     Method to train & evaluate a specified survival model on the data. 

#     Parameters
#     ----------
#     data_train: pd.DATAFRAME
#       Training dataframe preprocessed, filtered, undergone feature selection & additional dimensionality reduction (pycox models need data in this format)
#     data_test: pd.DATAFRAME
#       Test dataframe preprocessed, filtered, undergone feature selection & additional dimensionality reduction (pycox models need data in this format)
#     model: STRING
#       Model to train. Options include:
#       'CPH-L2' for Cox Proportional Hazards survival model with Ridge Penalty
#       'CPH-EN' for Cox Proportional Hazards survival model with elastic net penalty 
#       'CPH-GB' for Gradient Boosted unregularized Cox Proportional Hazards survival model
#       'CLS-GB' for Gradient Boosted Componentwise Least Squares survival model
#       'RSF' for Randomized Survival Forest survival model
#       'DEEP-SURV' for continuous time CoxPH / Deep Survival model [J. L. Katzman,et al (2018), https://arxiv.org/abs/1606.00931] - note this needs a different data input format, hyperparameter optimization, etc.
#     get_importances: INT
#        1 to get feature importance list of the model, 0 not to do so
#     verbose: INT
#        1 to print execution details, 0 not to do so
       
#     Returns
#     -------
#     surv_model: OBJECT (sksurv or pycox MODEL)
#     The trained model on the training dataset under the best hyperparameter configuration identified.
#     test_set_c_index: FLOAT
#     The Concordance Index attained by the model on the test set.
#     val_set_c_index: FLOAT
#     The Concordance Index attained by the model on the validation set.
#     training_set_c_index: FLOAT
#     The Concordance Index attained by the model on the training set.
#     importances_df: pd.DATAFRAME
#     Dataframe containing model information along with the model's feature importance list. Columns include 'Feature','Feature Importance', 'Feature Importance STD', 'Model', 'Test C-index', 'Validation C-index', 'Training C-index', 'Rank'.
    
#     """
        
#     #Prepare data for sksurv model format
#     # X_train, Y_train, X_test, Y_test = prepare_data_sksurv(data_train, data_test, target_cols)
#     return