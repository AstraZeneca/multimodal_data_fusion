import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from os.path import dirname, abspath, join
parent_of_filedir = dirname(dirname(abspath(__file__)))
sys.path.append(join(parent_of_filedir, 'utils'))
from data_utils import load_data, split_data, handle_missingdata, drop_missingdata
import feature_selection_utils


def get_features_per_modality(data, modalities_dict, feature_subset=None):
    """
    Method to identify the features in the data belonging to each modality.
    It reads data and modalities dictionary and return a dictionary whose keys are the different modalities and values
    are all features for that modality.

    Parameters
    ----------
    data: pd.DATAFRAME
      Dataframe containing data from 1 or more modalities
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features, in the format {'modality': 'suffix'}
    feature_subset: subset of features to be selected

    Returns
    -------
    features_per_modality: DICTIONARY 
      Dictionary listing all modalities included in the data and the list of features belonging to each, in the format {'modality': list_of_features}
    """
       
    # Feature columns per each modality identified based on their suffixes.
    features_per_modality = {}
    if feature_subset is None:
        for key, value in modalities_dict.items():
            features = [col for col in data.columns if col.endswith(value)]
            features_per_modality[key] = features
    else:
      i = 0
      for key, value in modalities_dict.items():
          if len(feature_subset[i]) > 0:
              features = [col for col in data.columns if col.endswith(value) and col in feature_subset[i]]
          else:
              features = [col for col in data.columns if col.endswith(value)]
          features_per_modality[key] = features

    return features_per_modality


def remove_low_variance(data_train, modalities_dict, subset, threshold_qconst, verbose):
    """
    Method to remove low variance features.
    It reads the training data, the modalities dictionary, the subset of modalities under investigation and a threshold.
    It returns the features in the modalities under investigation whose variance in the training set exceeds said threshold.

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Dataframe containing training data from 1 or more modalities
    cancer_type: LIST 
      List of strings specifying the cancer type(s) e.g. 'LUAD', 'LUSC', 'BRCA'
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features in the format {'modality': 'suffix'}
    subset: SET
      Set listing all modalities to be included in the analysis in the format {'modality_1', 'modality_2', ...}
    threshold_qconst: FLOAT
      Threshold below which features are consindered low variance and removed
    verbose: INT
      Flag; 1 to print detailed output, 0 to not do so
    
    Returns
    -------
    data_train.columns: LIST 
      List of all features in the modalities under investigation whose variance in the training set is above the threshold
    """
    # Isolate features per each modality:
    features_per_modality = get_features_per_modality(data_train, modalities_dict)
    
    features_to_retain = []
    for key, value in features_per_modality.items():
        if key in subset:#if modality is among the included ones in the experiment
            features_to_retain = features_to_retain + value
            
    data_train = data_train[features_to_retain]
    df = data_train.copy() #Make a working copy of the training set

    #Update list of features for each modality to account for removed columns:
    features_per_modality = get_features_per_modality(df, modalities_dict)
    print("Num features after selecting modality:")
    for key, value in features_per_modality.items():
        print(key+" : "+str(len(value)))
            
    # Identify quasi-constant (i.e. low-variance) features:
    qconstant_filter = VarianceThreshold(threshold=threshold_qconst)
    qconstant_filter.fit(df)

    qconstant_features_indices = np.where(qconstant_filter.get_support() == False)[0]
    qconstant_features = [df.columns[i] for i in qconstant_features_indices]
    if verbose == 1:
        print("There are " +str(len(qconstant_features))+ " features having the same value in about "+str((1-threshold_qconst)*100)+"% of the data:" )
        print(np.array(qconstant_features).T)

    #Remove quasi-constant (i.e. low-variance) features:
    data_train.drop(columns=qconstant_features, inplace=True)
    
    print("Dimensions of training dataset, after dropping low-variance features: "+str(data_train.shape))

    return data_train.columns

def remove_correlated(data_train, modalities_dict, subset, corr_threshold_high, verbose):
    """
    Method to remove correlated features from the dataset.
    It reads the training data, the modalities dictionary, the subset of modalities under investigation and a threshold.
    It identifies pairs of variables in the modalities under investigation whose Pearson correlation in the training set
    is above said threshold. It then returns a list of features not correlated with any other and the first feature
    (alphabetically) among correlated pairs.

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Dataframe containing training data from 1 or more modalities
    cancer_type: LIST 
      List of strings specifying the cancer type(s) e.g. 'LUAD', 'LUSC', 'BRCA'
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features in the format {'modality': 'suffix'}
    subset: SET
      Set listing all modalities to be included in the analysis in the format {'modality_1', 'modality_2', ...}
    corr_threshold_high: FLOAT
      Threshold above which features are consindered highly correlated (and 2nd from each pair is removed)
    verbose: INT
      Flag; 1 to print detailed output, 0 to not do so
    
    Returns
    -------
    data_train.columns: LIST 
      List of all features in the modalities under investigation whose correlation in the training set is below the threshold
    """
    #Isolate features of each modality:
    features_per_modality = get_features_per_modality(data_train, modalities_dict)
    
    features_to_retain = []
    for key, value in features_per_modality.items():
        if key in subset:#if modality is among the included ones in the experiment
            features_to_retain = features_to_retain + value
            
    data_train = data_train[features_to_retain] 
    df = data_train.copy() #Make a working copy of the training set
    
    col_corr = [] #List of all the names of deleted columns
    col_corr_ret = [] #List of retained columns in place of above (allows for repetitions)
    corr_matrix = df.corr()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= corr_threshold_high) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column to remove
                    col_corr.append(colname) 
                    colname_ret = corr_matrix.columns[j] # getting name of retained column highly correlated with i-th
                    col_corr_ret.append(colname_ret) 
                    if colname in df.columns:
                        del df[colname] # deleting the column from the dataset

    if verbose == 1:
        modality_list_string = "+".join(subset)
        print("There are "+ str(len(np.unique(col_corr)))+" highly correlated "+modality_list_string+" features with another one, having absolute pearson correlation > "+str(corr_threshold_high))
        if len(col_corr) > 0:
            print("The following "+ str(len(np.unique(col_corr)))+" are the ones deleted:")
            print(np.unique(col_corr))
            retained_list = np.unique(col_corr_ret)
            print("The following "+ str(len(retained_list))+" -correlated with the above- are retained:")   
            print(np.unique(col_corr_ret))
            print("More specifically:")
            for i in range(len(retained_list)):
                print("Feature "+retained_list[i]+" was retained in place of:")
                indices = np.where(np.array(col_corr_ret) == retained_list[i])
                print(np.array(list(col_corr))[indices])

    #Remove highly corellated features: #TODO: Refine which to keep? The one with highest abs correlation with target?    
    
    print("Dimensions of training dataset before dropping correlated columns: "+str(data_train.shape))
    data_train.drop(labels=col_corr, axis=1, inplace=True)
    print("Dimensions of training dataset after dropping correlated columns: "+str(data_train.shape))

    return data_train.columns

# TODO: Comment out for now .. needs debugging
def group_correlated(data_train, modalities_dict, subset, corr_threshold_high, verbose):
    """
    Method to group correlated features and return the average of each group.
    It reads the training data, the modalities dictionary, the subset of modalities under investigation and a threshold.
    It identifies groups of variables in the modalities under investigation whose Pearson correlation in the training set
    is above said threshold. It then replaces these features with the group's average. Finally, it returns a filtered
    version of the data which includes only features that are uncorrelated with others and averages of correlated groups
    (the group's name follows the convention 'group_'+name_of_1st_feature_in_group_in_alphabetical_order).

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Dataframe containing training data from 1 or more modalities
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features in the format {'modality': 'suffix'}
    subset: SET
      Set listing all modalities to be included in the analysis in the format {'modality_1', 'modality_2', ...}
    corr_threshold_high: FLOAT
      Threshold above which features are consindered highly correlated (and 2nd from each pair is removed)
    verbose: INT
      Flag; 1 to print detailed output, 0 to not do so
    
    Returns
    -------
    data_train: pd.DATAFRAME 
      Dataframe containing transformed training data in which correlated features are grouped & averaged
    """
    #Isolate features of each modality:
    features_per_modality = get_features_per_modality(data_train, modalities_dict)
    
    features_to_retain = []
    for key, value in features_per_modality.items():
        if key in subset:#if modality is among the included ones in the experiment
            features_to_retain = features_to_retain + value
            
    data_train = data_train[features_to_retain] 
    df = data_train.copy() #Make a working copy of the training set

    col_corr = [] #List of all the names of deleted columns
    col_corr_ret = [] #List of retained columns in place of above (allows for repetitions)
    corr_matrix = df.corr()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= corr_threshold_high) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column to remove
                    col_corr.append(colname) 
                    colname_ret = corr_matrix.columns[j] # getting name of retained column highly correlated with i-th
                    col_corr_ret.append(colname_ret) 
                    if colname in df.columns:
                        del df[colname] # deleting the column from the dataset

    if len(col_corr) > 0:
        retained_list = np.unique(col_corr_ret)
        for i in range(len(retained_list)):
            indices = np.where(np.array(col_corr_ret) == retained_list[i])
            correlated_group = [retained_list[i]] + list(np.array(list(col_corr))[indices])
            group_feature_name = 'group_'+min(correlated_group)#retained_list[i] #name group by 1st feature alphabetically
            data_train[group_feature_name] = data_train[correlated_group].mean(axis=1) #create a new feature for that modality, the mean of the correlated ones                        
            print("The following features formed a correlated group:") 
            print(correlated_group)
            print("They were all dropped and replaced by their group average: "+str(group_feature_name))

    #Remove highly corellated features: #TODO: Refine which to keep? The one with highest abs correlation with target?   
    
    print("Dimensions of training dataset before groupping correlated columns: "+str(data_train.shape))
    columns_to_drop = col_corr + col_corr_ret
    data_train.drop(labels=columns_to_drop, axis=1, inplace=True)
    print("Dimensions of training dataset after groupping correlated columns: "+str(data_train.shape))

    return data_train

def supervised_fs(data_train, target_vars, modalities_dict, subset, target_fs, method, K_top, discretization_bins, verbose):
    """
    Method to perform supervised feature selection. Currently supports information-theoretic methods but can extend to others.
    It reads the training data, the modalities dictionary, the subset of modalities under investigation and the parameters
    of feature selection: the feature selection method to be used, the number of most informative features to be returned,
    the number of discretization bins to be used to estimate information theoretic quantities and whether the target should
    be discretized or binarized. It identifies the K_top most informative variables with respect to the target based on the
    feature selection method. It then returns a list of the selected features' names.

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Dataframe containing training data from 1 or more modalities
    target_vars: LIST
      List of target variables, event and survival
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features in the format {'modality': 'suffix'}
    subset: SET
      Set listing all modalities to be included in the analysis in the format {'modality_1', 'modality_2', ...}
    target_fs: STRING
      Specifies whether the target is to be kept continuous, discretized using as many levels as the features, or binarized
      'discrete': Discretize target using as many equal width bins as features, for the purposes of feature selection
      'continuous': Use continuous target for the purposes of feature selection
      'median_binarized': Binarize target by comparing to median value, for the purposes of feature selection
    method: STRING
      Specifies which feature selection method is to be used. More can be added in the future. 
      For now the following information-theoretic methods are supported, among which JMI is the suggested: 
      'MIM': Univariate; ignores feature redundancy 
      'MIFS': Multivariate, but ignores feature complementarity
      'MRMR': Multivariate, but ignores feature complementarity
      'CIFE': Multivariate, accounts for relevance, redundancy & complementarity - red. is sum of pairwise associations (overvalued)
      'ICAP': Multivariate, as CIFE but w/o complementarity
      'DISR': Multivariate, as JMI, but normalized; more computationally expensive w/o any benefit
      'JMI': Multivariate, accounts for relevance, redundancy & complementarity - red. is average of pairwise associations (undervalued)
      'CMIM': Multivariate, accounts for relevance, redundancy & complementarity - red. is max of pairwise associations (undervalued less than JMI but penalizes complementarity)
    K_top: INT
      Number of most informative features to return
    discretization_bins: INT
      Number of bins to be used to discretize continuous variables for the purposes of calculating information theoretic quantities for feature selection     
    verbose: INT
      Flag; 1 to print detailed output, 0 to not do so
    
    Returns
    -------
    selected_feat: LIST
      List containing the names of the selected features
    """
        
    #Isolate features from different modalities:
    features_per_modality = get_features_per_modality(data_train, modalities_dict)#TODO: Refactor everything below!!!
    
    #Isolate feature values from different modalities:
    feature_values_per_modality = {}
    for key, value in features_per_modality.items():
        if key in subset:#if modality is among the included ones in the experiment
            feature_values_per_modality[key] = data_train[value].values
            
    #Isolate target(s):
    TARGET_ = data_train[target_vars[-1]].values # Isolate target(s) -- just use overall survival time (continuous)
    EVENT_ =  data_train[target_vars[0]].values # Event column (death) - binary (0/1); Cox univariate models need this
        
    #Discretize features for supervised feature selection:
    discretizer = preprocessing.KBinsDiscretizer(n_bins=discretization_bins, encode='ordinal')

    TARGET_disc = TARGET_
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        TARGET_disc = discretizer.fit_transform(TARGET_.reshape(-1, 1)).ravel()
    TARGET_= data_train[target_vars[-1]].values

    feature_values_per_modality_disc = feature_values_per_modality

    #Some clinical features need to be converted to numerical as they are categorical:
    if 'CLINICAL' in subset: #if 'CLINICAL' modality (clinical features) is among the included ones in the experiment
        for i in range(feature_values_per_modality['CLINICAL'].shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                enc = preprocessing.OrdinalEncoder()
                feature_values_per_modality_disc['CLINICAL'][:,i] = enc.fit_transform(feature_values_per_modality['CLINICAL'][:,i].reshape(-1, 1)).ravel().astype(int)                  

    for key in feature_values_per_modality_disc.keys():
        if key in subset and key != 'CLINICAL': #if modality is among the included ones in the experiment (and not CLINICAL)
            for i in range(feature_values_per_modality_disc[key].shape[1]):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    feature_values_per_modality_disc[key][:,i] = discretizer.fit_transform(feature_values_per_modality[key][:,i].reshape(-1, 1)).ravel()

    feat_names = []
    first_indx_modality = 0
    last_indx_modality = 0
    FEAT = np.zeros((data_train.shape[0], data_train.shape[1]-2))#-2 in dim 1, because TARGRET_ & EVENT are not included in the feature list
    for key, value in features_per_modality.items():
        if key in subset:#if modality is among the included ones in the experiment
            feat_names = feat_names + value
            last_indx_modality = last_indx_modality+len(value)
            FEAT[:, first_indx_modality:last_indx_modality] = feature_values_per_modality_disc[key]
            first_indx_modality = last_indx_modality
 
    if target_fs == 'discrete': #Discretize target using as many equal width bins as features, for the purposes of fs
        TARGET = TARGET_disc 
    elif target_fs == 'continuous': #Use continuous target for the purposes of fs
        TARGET = TARGET_
    elif target_fs == 'median_binarized': #Binarize target by comparing to median value, for the purposes of fs
        TARGET = (TARGET_ > np.median(TARGET_)).astype(np.int_) 
        
     #TODO: Allow for other methods (e.g. model-based: code is already available for this, just needs to be added - for now avoided to not overcomplicate things)  

    if method == 'MIM':#Univariate; ignores feature redundancy --results should be similar to own iplementation above
        feat_idx, J ,MIfy = feature_selection_utils.mim(FEAT, TARGET, n_selected_features=K_top)
    elif method == 'JMI':#Multivariate, accounts for relevance, redundancy & complementarity - red. is average of pairwise associations (undervalued)
        feat_idx, J ,MIfy = feature_selection_utils.jmi(FEAT, TARGET, n_selected_features=K_top)
    elif method == 'CMIM':# Multivariate, accounts for relevance, redundancy & complementarity - red. is max of pairwise associations (undervalued less than JMI but penalizes complementarity)    
        feat_idx, J ,MIfy = feature_selection_utils.cmim(FEAT, TARGET, n_selected_features=K_top)
    elif method == 'CORR':
        feat_idx = feature_selection_utils.correlation(FEAT, TARGET, n_selected_features=K_top)
    else:
      raise ValueError("Method not identified")
    
    n_feat = min(FEAT.shape[1], K_top) 
    selected_feat = [feat_names[i] for i in feat_idx[:n_feat]]

    if verbose == 1:
        if method == 'MIM':
            print("Top "+str(n_feat)+" individually most informative features of the target under "+method+" (ranked in descending order):")
        else:
            print("Top "+str(n_feat)+" jointly most informative features of the target under "+method+" (ranked in descending order):")
        print(np.array(selected_feat[:n_feat]).T)

    return selected_feat

def aggregate_bootstrap_fs(selected_featureset, consistency_threshold, verbose):
    """
    Method to aggregate the feature sets selected under the various bootstraps.
    It reads the selected feature set per each bootstrap and a consistency threshold (on the number of bootstraps).
    It returns the features selected in at least as many bootstraps as specified by the consistency threshold.  

    Parameters
    ----------
    selected_featureset: LIST
      List of lists; the n-th sublist contains the names of the features selected on the n-th bootstrap 
    consistency_threshold: INT
      Minimal number of bootstraps in which a feature must appear to be selected
    verbose: INT
      Flag; 1 to print detailed output, 0 to not do so
    
    Returns
    -------
    selected_feat: LIST
      List containing the names of the selected features
    """
    
    n_feat = len(selected_featureset[0]) 
    union = []
    for i in range(len(selected_featureset)):
        union = list(set(union) | set(selected_featureset[i]))

    print("The "+str(len(selected_featureset))+" selected feature sets of up to "+str(n_feat)+" features each resulted in a total of "+str(len(union))+" selected features.")

    occurences = np.zeros((len(union), 1))
    for i in range(len(union)):
        for j in range(len(selected_featureset)):
            if union[i] in selected_featureset[j]:
                occurences[i] = occurences[i] + 1

    sorted_indices=(-occurences.ravel()).argsort()#order features by frequency of appearence in a featureset
    
    if verbose == 1:
        print("Feature / Occurences in "+str(len(selected_featureset))+" selected feature sets")
        for i in range(len(union)):
            print(union[sorted_indices[i]]+" / "+str(int(occurences[sorted_indices[i]])))

    #Aggregate
    if occurences[sorted_indices][n_feat-1] > consistency_threshold-1:#If all features are selected consistently enough
        last_indx = n_feat # will include them all
    else:
        last_indx = list(occurences[sorted_indices]).index(consistency_threshold-1) #Find index of sorted features below which occurences are below the threshold
   
    print("FINAL LIST OF SELECTED FEATURES:")
    for i in range(last_indx):
        print(union[sorted_indices[i]]+" / "+str(int(occurences[sorted_indices[i]])))

    #TODO: Aggregate bootstrap *rankings* using social choice theory methods, not just selected sets    

    #Store final feature list:
    selected_feat = [union[i] for i in sorted_indices[:last_indx]]

    return selected_feat

def feature_selection_main(data_train, target_vars, modalities_dict, subset,
                           drop_qconstant, threshold_qconst, handle_correlated, corr_threshold_high,
                           target_fs, method, K_top, discretization_bins,
                           consistency_threshold, save_selected, verbose):
    """
    Main method of the feature selection pipeline.
    It reads the training data, the modalities dictionary, the subset of modalities under investigation and the parameters
    of unsupervised & supervised feature selection. It returns the set of selected features after all feature selection steps.

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Dataframe containing training data from 1 or more modalities
    target_vars: LIST
      List of target variables, example - censor and OS
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features in the format {'modality': 'suffix'}
    subset: SET
      Set listing all modalities to be included in the analysis in the format {'modality_1', 'modality_2', ...}
    drop_qconstant: 'INT'  
      Flag; '1' to drop low variance features, '0' to not do so
    threshold_qconst: FLOAT
      Threshold below which features are consindered low variance and removed
    handle_correlated: STRING
      Specifies how to handle correlated features
      'drop' to remove correlated features
      'group' to average them
      'ignore' to not do anything about them
    corr_threshold_high: FLOAT
      Threshold above which features are consindered highly correlated (and 2nd from each pair is removed)
    target_fs: STRING
      Specifies whether the target is to be kept continuous, discretized using as many levels as the features, or binarized
      'discrete': Discretize target using as many equal width bins as features, for the purposes of feature selection
      'continuous': Use continuous target for the purposes of feature selection
      'median_binarized': Binarize target by comparing to median value, for the purposes of feature selection
    method: STRING
      Specifies which feature selection method is to be used. More can be added in the future. 
      For now the following information-theoretic methods are supported, among which JMI is the suggested: 
      'MIM': Univariate; ignores feature redundancy 
      'MIFS': Multivariate, but ignores feature complementarity
      'MRMR': Multivariate, but ignores feature complementarity
      'CIFE': Multivariate, accounts for relevance, redundancy & complementarity - red. is sum of pairwise associations (overvalued)
      'ICAP': Multivariate, as CIFE but w/o complementarity
      'DISR': Multivariate, as JMI, but normalized; more computationally expensive w/o any benefit
      'JMI': Multivariate, accounts for relevance, redundancy & complementarity - red. is average of pairwise associations (undervalued)
      'CMIM': Multivariate, accounts for relevance, redundancy & complementarity - red. is max of pairwise associations (undervalued less than JMI but penalizes complementarity)
    K_top: INT
      Number of most informative features to return
    discretization_bins: INT
      Number of bins to be used to discretize continuous variables for the purposes of calculating information theoretic quantities for feature selection
    consistency_threshold: INT
      Minimal number of bootstraps in which a feature must appear to be selected
    save_selected: INT
      Flag; 1 to save training & test sets containing only selected features, 0 to not save them
    verbose: INT
      Flag; 1 to print detailed output, 0 to not do so

    Returns
    -------
    selected_feat: LIST
      List containing the names of the selected features
    """
    modality_list_string = "+".join(subset)
    print("WORKING ON MODALITY: "+modality_list_string)

    #-----Removing low variance features:-----
    if drop_qconstant == 1:
        print("REMOVING LOW VARIANCE FEATURES")
        
        #Working on features only:
        data_train_ufs = data_train.drop(target_vars, axis=1)
        
        #Find features to retain:
        features_ = remove_low_variance(data_train_ufs, modalities_dict, subset, threshold_qconst, verbose)
        
        #Update training set:
        data_train_ufs = data_train[list(features_) + target_vars]
      
        #Update list of features for each modality to account for any removed columns:
        features_per_modality = get_features_per_modality(data_train_ufs, modalities_dict)
        print("Num features after dropping zero/low variance:")
        for key, value in features_per_modality.items():
            print(key+" : "+str(len(value)))

    #-----Removing correlated features:-----
    if handle_correlated == 'drop':
        print("REMOVING CORRELATED FEATURES")
        
        #Working on features only:
        data_train_ufs = data_train.drop(target_vars, axis=1)
        
        #Update training set:
        features_ = remove_correlated(data_train_ufs, modalities_dict, subset, corr_threshold_high, verbose)
        
        #Update training set:
        data_train_ufs = data_train[list(features_) + target_vars]
   
        #Update list of features for each modality to account for any removed columns:
        features_per_modality = get_features_per_modality(data_train_ufs, modalities_dict)
        print("Num features after dropping corellated:")
        for key, value in features_per_modality.items():
            print(key+" : "+str(len(value)))
    
    #-----Grouping correlated features:-----
    elif handle_correlated == 'group': #TODO: BUG HERE -FIX!
        print("GROUPING CORRELATED FEATURES")
        
        #Working on features only:
        data_train_ufs = data_train.drop(target_vars, axis=1)
        
        #Update feature set:
        data_train_ufs = group_correlated(data_train_ufs, modalities_dict, subset, corr_threshold_high, verbose)

        #Merge target columns to updated feature set:
        data_train_ufs = pd.concat([data_train_ufs, data_train[target_vars]], axis=1) 
        
        #Update list of features for each modality to account for any removed columns:
        features_per_modality = get_features_per_modality(data_train_ufs, modalities_dict)
        print("Num features after groupping corellated:")
        for key, value in features_per_modality.items():
            print(key+" : "+str(len(value)))

    else: # If no unsupervised feature selection is performed
        data_train_ufs = data_train
        
    #----Selecting Features based on MI/correlation/univariate Cox models with target----
    print("SELECTING INFORMATIVE FEATURES OF TARGET")
    
    #Using the result of unsupervised feature selection:
    data_train = data_train_ufs
    
    # Call main supervised feature selection method:
    selected_feat = supervised_fs(data_train, target_vars, modalities_dict, subset, target_fs, method, K_top, discretization_bins, verbose)

    return selected_feat

def feature_selection_pipeline(data, cancer_type, modalities_dict, subset, random_split, include_PDL1_status, modality_handling_mode,
                               subset_selection_mode, preprocess_modality_dict, frac_train, split_mode, data_char_dict,
                               unsupervised_fs_options_dict, supervised_fs_options_dict, fs_consistency_options_dict, user_options_dict, stratify=False, strat_cols = []):
    """
    Main method of the entire pipeline. Calls all other methods for preprocessing data & performing feature selection.
    
    Parameters
    ----------
    data: LIST
      Data in the form of a list of dataframes
    cancer_type: LIST 
      List of strings specifying the cancer type(s) e.g. 'LUAD', 'LUSC', 'BRCA'
    excluded_cancer_stage: SET
      Set of cancer stages to exclude from analysis; can be empty
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features in the format {'modality': 'suffix'}
    subset: SET
      Set listing all modalities to be included in the analysis in the format {'modality_1', 'modality_2', ...}
    random_split: INTEGER
      Split number for repeatability
    include_PDL1_status: BOOL
      Flag: TRUE to include PDL1 status in the models; FALSE not to do so
    modality_handling_mode: STRING
      Specifies if feature selection steps are going to be applied to all modalities jointly or to each separately
      'per_modality': For intermediate fusion
      'joint': For early fusion
    subset_selection_mode: STRING
      Specifies how to select features
      'domain_knowledge': Only select features based on BIKG
      'feature_selection': Only select features based on supervised feature selection
      'mixed': Do both of the above
      'none': Do not do any feature selection (can still drop low variance / correlated based on 'unsupervised_fs_options_dict' options)
    preprocess_modality_dict: DICTIONARY
      Options regarding the preprocessing of each modality Format is  {'modality': 'normalization'} ; 'normalization' is any of 'standardize', 'robust_scale' or 'minmax_scale' for continuous features
    frac_train: FLOAT
      Fraction of the data to be used for training 
    split_mode: STRING
      Specifies if we split the data per row ('row'), or per subject ID ('subjid') - a fraction of frac_train of rows / subjects will be used for training and the rest for testing
    data_char_dict: DICTIONARY
      Dictionary with data column labels
    unsupervised_fs_options_dict: DICTIONARY
      Dictionary with unsupervised feature selection parameters as keys: 'drop_qconstant', 'threshold_qconst, 'handle_correlated', 'corr_threshold_high'
    supervised_fs_options_dict: DICTIONARY
      Dictionary with unsupervised feature selection parameters as keys: 'discretization_bins', 'target_fs', 'method', 'K_top'
    fs_consistency_options_dict: DICTIONARY
      Dictionary with feature selection consistenct parameters as keys: 'n_runs', 
    user_options_dict: DICTIONARY
      Various user options; keys include 'verbose', 'save_selected' and 'consistency_threshold'
    stratify: BOOL
      Dataset stratification option
    strat_col: LIST
      List of columns to base startification upon.

    Returns
    -------
    data_train: pd.DATAFRAME
      Training dataframe preprocessed, filtered, undergone feature selection for downstream analysis
    data_test: pd.DATAFRAME
      Test dataframe preprocessed, filtered, undergone feature selection for downstream analysis
    """
        
    #Unpack arguments:
    drop_qconstant = unsupervised_fs_options_dict['drop_qconstant']
    threshold_qconst = unsupervised_fs_options_dict['threshold_qconst']
    handle_correlated = unsupervised_fs_options_dict['handle_correlated']
    corr_threshold_high = unsupervised_fs_options_dict['corr_threshold_high']

    discretization_bins = supervised_fs_options_dict['discretization_bins']
    target_fs = supervised_fs_options_dict['target_fs']
    method = supervised_fs_options_dict['method']
    K_top = supervised_fs_options_dict['K_top']
    
    n_runs = fs_consistency_options_dict['n_runs']
    consistency_threshold = fs_consistency_options_dict['consistency_threshold']

    verbose = user_options_dict['verbose']
    save_selected = user_options_dict['save_selected']

    target_vars = data_char_dict['targets']
    subjid_col = data_char_dict['subject']
    pdl1_col = data_char_dict['pdl1']
    missingdata_mode = data_char_dict['missingdata_mode']
    feature_subset = data_char_dict['features']

    assert len(modalities_dict.keys()) == len(feature_subset), "Features for each modality needs to be indicated"

    # data_ = data[0]
    # If shorthand 'ALL' was used for cancer type, expand:
    if cancer_type == ['ALL']: 
        cancer_type = ['ACC', 'BLCA', 'DLBC', 'UCEC', 'SKCM', 'HNSC', 'PRAD', 'KIRP',
                       'PAAD', 'SARC', 'CESC', 'COAD', 'LUSC', 'READ', 'KIRC', 'LIHC',
                       'BRCA', 'OV', 'UCS', 'GBM', 'KICH', 'THCA', 'LGG', 'LUAD', 'MESO',
                       'PCPG', 'TGCT', 'UVM', 'THYM', 'CHOL', 'ESCA', 'STAD', 'LAML']
        
    # Choose data of only the selected cancer type (store in a new dataframe data_)
    data_ = data[0].query('type in @cancer_type').reset_index(drop=True).copy()
    
    print(f"{data_.shape[0]} patients of the following CANCER TYPES are INCLUDED: {cancer_type}")
    # print(cancer_type)
    
    # Isolate features per each modality:
    features_per_modality = get_features_per_modality(data_, modalities_dict, feature_subset)
    
    # Names of target (event & time-to-event) columns:
    CLIN_COLS_TARG = target_vars #OS (overall survival) denotes event (death) & OS.time (overall survival time) are targets

    dropRows = False
    if "drop" in missingdata_mode:
      dropRows = True
    # if exists, use pre-defined splits
    if len(data) > 1:
        data_train, data_test = data
        if dropRows:
            data_train = drop_missingdata(data=data_train, cols=features_per_modality, mode=missingdata_mode)
            data_test = drop_missingdata(data=data_test, cols=features_per_modality, mode=missingdata_mode)
    else:
        data = data[0]
        if dropRows:
            data = drop_missingdata(data=data, cols=features_per_modality, mode=missingdata_mode)
        if stratify:
            data_train = data.groupby(strat_cols, group_keys=False).sample(frac=frac_train, random_state=random_split)
            data_test =  pd.concat([data, data_train]).drop_duplicates(keep=False)
        else:
            if split_mode == 'row':
                print('Train/Test split PER ROW - Unstratified.')
                data_train, data_test = train_test_split(data_, test_size=1-frac_train, random_state=random_split)
            else: # split_mode == 'subjid'
                print('Train/Test split PER SUBJECT - Unstratified.')
                import random
                
                unique_SUBJIDs = data_[subjid_col].unique()
                random.Random(random_split).shuffle(unique_SUBJIDs) # Make sure in each split we are using different subjects in each subset
                train_size = int(frac_train * len(unique_SUBJIDs))
                print('Splitting per subject results in including '+str(train_size)+' subjects in the training set & '+str(len(unique_SUBJIDs)-train_size)+' subjects in the test set.' )
                train_SUBJIDs = unique_SUBJIDs[:train_size]
                test_SUBJIDs = unique_SUBJIDs[train_size:]
                data_train = data_.loc[data_[subjid_col].isin(train_SUBJIDs)]
                data_test = data_.loc[data_[subjid_col].isin(test_SUBJIDs)]
    # Split training data into train/val
    print('Train/Val split ')
    data_train, data_val = train_test_split(data_train, test_size=1-frac_train, random_state=random_split)  
    print("SUBSET OF INTEREST IDENTIFIED & TRAIN/TEST SPLIT PERFORMED")
    print("Dimensions of full dataset: "+str(data_.shape))
    print("Dimensions of training dataset: "+str(data_train.shape))
    print("Dimensions of validation dataset: "+str(data_val.shape))
    print("Dimensions of test dataset: "+str(data_test.shape))

    #Choose only features from selected modalities:
    selected_column_names = []
    for key, value in features_per_modality.items():
        if key in subset: #if modality is among the included ones in the experiment
            selected_column_names = selected_column_names + value 
    
    # Handle missing data
    data_train, imputation_vals = handle_missingdata(data=data_train, cols=features_per_modality, mode=missingdata_mode)
    data_test, _ = handle_missingdata(data=data_test, cols=features_per_modality, mode=missingdata_mode, 
                                      column_vals=imputation_vals)
    
    #Store subject ID per training & test datapoint. Might be used in the future:
    SUBJID_TR = data_train[subjid_col]
    SUBJID_VAL = data_val[subjid_col]
    SUBJID_TE = data_test[subjid_col]
    
    #Store PDL1 status per training & test datapoint. Might be used in the future:
    if include_PDL1_status:
      PDL1_TR = data_train[pdl1_col]
      PDL1_VAL = data_val[pdl1_col]
      PDL1_TE = data_test[pdl1_col]
    
    ######################TODO: Add imputation?  
    
    ###########################################
    
    # FEATURE SELECTION (UNSUPERVISED & SUPERVISED, BASED ON DOMAIN KNOWLEDGE OR STATISTICAL):
    print("FEATURE SELECTION INITIATED.")
    
    if subset_selection_mode == 'none':# if no dimensionality reduction is to be applied to the chosen modality/ies:
        
        data_train = data_train[selected_column_names + CLIN_COLS_TARG]
        data_val = data_val[selected_column_names + CLIN_COLS_TARG]
        data_test = data_test[selected_column_names + CLIN_COLS_TARG]   
        
        features_per_modality = get_features_per_modality(data_train, modalities_dict)
        print("Num features after using all features of chosen modality/ies:")
        for key, value in features_per_modality.items():
            print(key+" : "+str(len(value)))
        
    else: #if selecting features or combining feature selection to domain knowledge chosen ones:      
        # Preprocess features 
        ###########################  TODO: Might move to preprocessing
        for key, value in features_per_modality.items(): #for each modality
            if key in subset: #if modality is among the included ones in the experiment
                if key in preprocess_modality_dict['modality']:  #if a preprocessing step is specified for said modality
                    indx = preprocess_modality_dict['modality'].index(key)
                    if preprocess_modality_dict['type'][indx] == 'standardize': #if said step is standardization (i.e. subtract mean and scales by standard deviation)
                        sc = preprocessing.StandardScaler()
                        data_train[value] = sc.fit_transform(data_train[value])
                        data_val[value] = sc.transform(data_val[value])
                        data_test[value] = sc.transform(data_test[value])
                    elif preprocess_modality_dict['type'][indx] == 'minmax_scale': #if said step is minmax scaling (i.e. to lie within [0, 1])
                        sc = preprocessing.MinMaxScaler()
                        data_train[value] = sc.fit_transform(data_train[value])
                        data_val[value] = sc.transform(data_val[value])
                        data_test[value] = sc.transform(data_test[value])
                    elif preprocess_modality_dict['type'][indx] == 'robust_scale': #if said step is robust scaling (i.e. subtract median and scales by interquantile range)
                        sc = preprocessing.RobustScaler()
                        data_train[value] = sc.fit_transform(data_train[value])
                        data_val[value] = sc.transform(data_val[value])
                        data_test[value] = sc.transform(data_test[value])
        ##########################
            
        #Get training set bootstrap samples 
        data_train_orig = data_train[selected_column_names + CLIN_COLS_TARG]
        data_val_orig = data_val[selected_column_names + CLIN_COLS_TARG]
        data_test_orig = data_test[selected_column_names + CLIN_COLS_TARG]
        bootstrap_size = data_train_orig.shape[0] #each bootstrap is of size equal to the original training set
        selected_featureset = [] #This will be the list of the n_runs selected featuresets (1 under each bootstrap)

        for run in range(np.max([1, n_runs])): # Run at least once -- if bootstrapping is to be applied will run for more
            # Check if bootstrapping is to be applied:
            if n_runs>0: 
                data_train_bootstrap = data_train_orig.sample(bootstrap_size, replace=True)# Treat the bootstrap sample of the training set as the current training set
                print("BOOTSTRAP "+str(run+1)+" OF "+str(n_runs))  
            else: #If n_runs == 0, no bootstrapping is performed.
                print("FEATURE SELECTION ON ENTIRE TRAINING SET - NO BOOTSTRAPPING PERFORMED!")
                data_train_bootstrap = data_train_orig.copy() #'bootstrap' is just full training set
            
            #Check if we need to run seperately per each modality or in joint/single modality mode:
            if (modality_handling_mode == 'joint') or (len(subset)==1): # If single modality examined or all modalities jointly:
                data_train = data_train_bootstrap#.drop(['OS'], axis=1) #TODO: Remove this. Ensure no shuffling occurs, then move inside feature_selection_main as a first step, and undo it before supervised_fs
                selected_feat = feature_selection_main(data_train, target_vars, modalities_dict, subset,
                                                       drop_qconstant, threshold_qconst, handle_correlated, corr_threshold_high,
                                                       target_fs, method, K_top, discretization_bins,
                                                       consistency_threshold, save_selected, verbose)

            elif (modality_handling_mode == 'per_modality'):# If features are to be selected per modality:
                print("INITIATING MODALITY-WISE FEATURE SELECTION")
                selected_feat = []
                for key, value in features_per_modality.items():
                    if key in subset: #if modality is among the included ones in the experiment
                        data_train_modality = data_train_bootstrap[value + target_vars] #A modality-wise version of the training data (bootstrap)
                        selected_feat_modality = feature_selection_main(data_train_modality, target_vars, modalities_dict, {key},
                                                                        drop_qconstant, threshold_qconst, handle_correlated, corr_threshold_high,
                                                                        target_fs, method, K_top, discretization_bins,
                                                                        consistency_threshold, save_selected, verbose)
                        selected_feat = selected_feat + selected_feat_modality
             
                
            else: #Invalid combination of modality_handling_mode & subset
                print('Invalid value for modality_handling_mode &/or subset. Terminating.')

            selected_featureset.append(np.array(selected_feat).T)

        #If several featuresets produced, aggregate them:
        if n_runs > 1:
            print("AGGREGATING FEATURES FROM ALL BOOTSTRAPS")
            selected_feat = aggregate_bootstrap_fs(selected_featureset, consistency_threshold, verbose)

        selected_column_names = np.concatenate([selected_feat, CLIN_COLS_TARG], axis=0)
        
        data_train = data_train_orig[selected_column_names]#careful here: use original training set -not bootstrap- to get correct targets
        data_val = data_val_orig[selected_column_names]
        data_test = data_test_orig[selected_column_names]

        #Report number of selected features per modality:
        features_per_modality = get_features_per_modality(data_train, modalities_dict)
        print("Num features after choosing the most consistently selected:")
        for key, value in features_per_modality.items():
            print(key+" : "+str(len(value)))
            
    #Save datasets:
    if save_selected == 1: #Do not save featureset selected for random targets
        modality_list_string = "_".join(subset)
    

        data_train.to_parquet('./filtered_data_train_.parquet')
        data_val.to_parquet('./filtered_data_val_.parquet')
        data_test.to_parquet('./filtered_data_test_.parquet')
    
    # TODO: Add pdl1 or remove based on requirement (and absence in feature selection)
    #Add PDL1 status, even if model uses no other clinical features, if user specifically requested so:
    if include_PDL1_status and 'CLINICAL' not in subset:
        data_train = pd.concat([data_train, PDL1_TR], axis=1)
        data_val = pd.concat([data_val, PDL1_VAL], axis=1) 
        data_test = pd.concat([data_test, PDL1_TE], axis=1)
        print('PDL1 STATUS INCLUDED BY USER REQUEST')
    #Remove PDL1 status, even if model uses the other clinical features, if user specifically requested so:
    elif not include_PDL1_status and 'CLINICAL' in subset and pdl1_col in data_train.columns:
        data_train = data_train.drop(pdl1_col, axis = 1)
        data_val = data_val.drop(pdl1_col, axis = 1)
        data_test = data_test.drop(pdl1_col, axis = 1)
        print('PDL1 STATUS EXCLUDED BY USER REQUEST')
    else:
        print('PDL1 STATUS RECEIVED SAME TREATMENT AS ALL OTHER CLINICAL FEATURES')
    
    #Report number of selected features per modality:
    features_per_modality = get_features_per_modality(data_train, modalities_dict)
    print("Num features to be used for model training:")
    for key, value in features_per_modality.items():
        print(key+" : "+str(len(value)))
    
    print("FEATURE SELECTION FINISHED!")
        
    return(data_train, data_val, data_test) # Returns train, val & test set including only selected features (features were selected on this training set)

  
def additional_dimensionality_reduction(data_train, data_test, additional_dr_options, modalities_dict, subset, verbose):
    """
    Method to call and train an autoencoder of prespecified hyperparameters on the data. 

    Parameters
    ----------
    data_train: pd.DATAFRAME
      Training dataframe preprocessed, filtered, undergone feature selection, before additional dimensionality reduction
    data_test: pd.DATAFRAME
      Test dataframe preprocessed, filtered, undergone feature selection, before additional dimensionality reduction
    additional_dr_options: DICTIONARY
      Hyperparameters of the additional dimensionality reduction;
      For now, key 'model' supports only 'autoencoder' or 'none', i.e no additional dimensionality reduction is applied.
      For 'autoencoder', keys are 'FRAC_TRAIN_AE', 'HIDDEN_DIM', 'LATENT_SPACE', 'EPOCHS', 'BATCH_SIZE' & 'ES_PATIENCE'
    modalities_dict: DICTIONARY 
      Dictionary listing all modalities included in the data and the suffixes identifying their corresponding features, in the format {'modality': 'suffix'}
    subset: SET
      Set listing all modalities to be included in the analysis in the format {'modality_1', 'modality_2', ...}
    verbose: INT
       1 to print execution details, 0 not to do so
      
    Returns
    -------
    encoder Object
        Trained keras.MODEL encoder if 'model' is 'autoencoder' or None object if 'model' is 'none'
    data_train_processed: pd.DATAFRAME
      Training dataframe preprocessed, filtered, undergone feature selection, after additional dimensionality reduction
    data_test_processed: pd.DATAFRAME
      Test dataframe preprocessed, filtered, undergone feature selection, after additional dimensionality reduction
    """
    
    if additional_dr_options['model'] == 'autoencoder':
        print("AUTOENCODER RUNNING")
        raise ValueError("Autoencoder currently not implemented")
        #Unpack AE Hyperparameters: 
        # FRAC_TRAIN_AE = additional_dr_options['model_hyperparameters']['FRAC_TRAIN_AE']
        # HIDDEN_DIM = additional_dr_options['model_hyperparameters']['HIDDEN_DIM']
        # LATENT_SPACE = additional_dr_options['model_hyperparameters']['LATENT_SPACE']
        # EPOCHS = additional_dr_options['model_hyperparameters']['EPOCHS']
        # BATCH_SIZE = additional_dr_options['model_hyperparameters']['BATCH_SIZE']
        # ES_PATIENCE = additional_dr_options['model_hyperparameters']['ES_PATIENCE']

        # #Get original training & test data:
        # X_train_orig, Y_train, X_test_orig, Y_test = prepare_data_sksurv(data_train, data_test)

        # if additional_dr_options['modality_handling_mode_dr'] == 'joint':
        #     print('RUNNING AUTOENCODER IN JOINT MODALITY MODE')

        #     unchanged_modalities_count = 0 # Counts number of modalities whose features are not transformed
            
        #     #Find modalities whose features are to not be transformed and store them as they are:
        #     for modality in subset:
        #         if modality in additional_dr_options['excluded_modalities']:
        #             print('USER REQUESTED FEATURES FROM MODALITY '+modality+' TO NOT BE TRANSFORMED BY THE AUTOENCODER; KEEPING THESE FEATURES AS THEY ARE')
        #             unchanged_list = X_train_orig.columns[pd.Series(X_train_orig.columns).str.endswith(modalities_dict[modality])]
        #             if verbose == 1:
        #                 print(unchanged_list)
        #             if unchanged_modalities_count == 0: # if this is the first such modality, initialize list of unchanged:
        #                 X_train_unchanged = X_train_orig[unchanged_list]
        #                 X_test_unchanged = X_test_orig[unchanged_list]
        #                 X_train_to_transform = X_train_orig.drop(columns=unchanged_list)
        #                 X_test_to_transform = X_test_orig.drop(columns=unchanged_list)
        #                 unchanged_modalities_count = unchanged_modalities_count + 1
        #             else: #for any subsequent excluded modality:
        #                 X_train_unchanged = pd.concat([X_train_unchanged, X_train_orig[unchanged_list]], axis=1)
        #                 X_test_unchanged = pd.concat([X_test_unchanged, X_test_orig[unchanged_list]], axis=1) 
        #                 X_train_to_transform = X_train_orig.drop(columns=unchanged_list)
        #                 X_test_to_transform = X_test_orig.drop(columns=unchanged_list)
        #                 unchanged_modalities_count = unchanged_modalities_count + 1
                    
        #             if verbose == 1:
        #                 print('************ORIGINAL DATA,  EXCLUDED MODALITIES)**************')
        #                 print('Number of excluded modalities: '+str(unchanged_modalities_count))
        #                 print('Feature List:')
        #                 print(X_test_unchanged.columns)
        #                 print('X Test dimensions:')
        #                 print(X_test_unchanged.shape)
        #                 print('X Train dimensions:')
        #                 print(X_train_unchanged.shape)   
                        
            
        #     #If there are no modalities to exclude from autoencoder, use original (full feature set), otherwise only use the included ones (already set):
        #     if unchanged_modalities_count == 0:
        #         X_train_to_transform = X_train_orig
        #         X_test_to_transform = X_test_orig
                
        #     #Train the autoencoder:
        #     encoder, decoder, history = train_autoencoder(X_train_to_transform, FRAC_TRAIN_AE, HIDDEN_DIM, LATENT_SPACE, EPOCHS,BATCH_SIZE, ES_PATIENCE)

        #     #Transform features using the encoder:
        #     X_train = encode(X_train_to_transform, encoder)
        #     X_test = encode(X_test_to_transform, encoder)
            
        #     #Add suffix indicating latent variables are derived from mixed modalities:
        #     X_train = X_train.add_suffix('_mixed')
        #     X_test = X_test.add_suffix('_mixed')
            
        #     #Merge with unchanged modalities, if any specified:
        #     if unchanged_modalities_count > 0:
        #         X_train = pd.concat([X_train, X_train_unchanged], axis=1)
        #         X_test = pd.concat([X_test, X_test_unchanged], axis=1)

        # else: #additional_dr_options['modality_handling_mode_dr'] == 'per_modality'
        #     print('RUNNING AUTOENCODER IN PER-MODALITY MODE')

        #     first_modality_flag = 1 # Flag used to initialize the dataframes to be returned (on 1st modality)
                
        #     features_per_modality = get_features_per_modality(data_train, modalities_dict)
            
        #     for key, value in features_per_modality.items():
        #         if key in subset: #if modality is among the included ones in the experiment

        #             print('PROCESSING MODALITY: '+key)
        #             X_train_modality_orig = X_train_orig[value] #A modality-wise version of X_train
        #             X_test_modality_orig = X_test_orig[value] #A modality-wise version of X_test 
                    
        #             if verbose == 1:
        #                 print('************ORIGINAL DATA,  MODALITY: '+key+'**************')
        #                 print('Feature List:')
        #                 print(X_test_modality_orig.columns)
        #                 print('X Test dimensions:')
        #                 print(X_test_modality_orig.shape)
        #                 print('X Train dimensions:')
        #                 print(X_train_modality_orig.shape)
                        
        #             #Train the autoencoder on selected modality:
        #             encoder, decoder, history = train_autoencoder(X_train_modality_orig, FRAC_TRAIN_AE, HIDDEN_DIM, LATENT_SPACE, EPOCHS,BATCH_SIZE, ES_PATIENCE)

        #             #Transform features of selected modality using the modality-specific encoder:
        #             if key in additional_dr_options['excluded_modalities']:#...if processing features from a modality that is excluded from the autoencoder
        #                 #Use the unchanged features:
        #                 print('USER REQUESTED FEATURES FROM MODALITY '+key+' TO NOT BE TRANSFORMED BY THE AUTOENCODER; KEEPING THESE AS THEY ARE')
        #                 X_train_modality = X_train_orig[value]
        #                 X_test_modality = X_test_orig[value]
        #             else:
        #                 #In any other case, transform features:
        #                 print('FEATURES FROM MODALITY '+key+' ARE TRANSFORMED BY THE AUTOENCODER')
        #                 X_train_modality = encode(X_train_modality_orig, encoder)
        #                 X_test_modality = encode(X_test_modality_orig, encoder)
            
        #                 #Add modality suffix:
        #                 X_train_modality = X_train_modality.add_suffix(modalities_dict[key]) #'_'+key.lower()
        #                 X_test_modality = X_test_modality.add_suffix(modalities_dict[key]) #'_'+key.lower()
                    
        #             if verbose == 1:
        #                 print('************TRANSFORMED DATA,  MODALITY: '+key+'**************')
        #                 print('Feature List:')
        #                 print(X_test_modality.columns)
        #                 print('X Test dimensions: ')
        #                 print(X_test_modality.shape)
        #                 print('X Train dimensions: ')
        #                 print(X_train_modality.shape)

                
        #             #Aggregate extracted features from each modality:
        #             if first_modality_flag == 1: # Check if initialization is needed (first modality)
        #                 X_train = X_train_modality
        #                 X_test = X_test_modality
        #                 first_modality_flag = 0
        #             else:#For any subsequent modality added, just grow the dataframe
        #                 X_train = pd.concat([X_train, X_train_modality], axis=1)
        #                 X_test = pd.concat([X_test, X_test_modality], axis=1)
                        
        # if verbose == 1:
        #     print('************TRANSFORMED DATA, ALL MODALITIES (FINAL):**************')
        #     print('Feature List:')
        #     print(X_test.columns)
        #     print('X Test dimensions')
        #     print(X_test.shape)
        #     print('X Train dimensions')
        #     print(X_train.shape)
        #     print('Y Test dimensions')
        #     print(Y_test.shape)
        #     print('Y Train dimensions')
        #     print(Y_train.shape)
            
        # # Merge targets and transformed features for downstream analysis:
        # events_train, durations_train = zip(*Y_train)
        # data_train_processed = X_train
        # data_train_processed[target_vars[0]] = events_train
        # data_train_processed[target_vars[0]] = data_train_processed[target_vars[0]].astype(int).astype(float) # Converting Boolean to 1.0 or 0.0
        # data_train_processed[target_vars[-1]] = durations_train
        # data_train_processed[target_vars[-1]] = data_train_processed[target_vars[-1]].astype(int) # Convert survival times to integers
        # events_test, durations_test = zip(*Y_test)
        # data_test_processed = X_test
        # data_test_processed[target_vars[0]] = events_test
        # data_test_processed[target_vars[0]] = data_test_processed[target_vars[0]].astype(int).astype(float) # Converting Boolean to 1.0 or 0.0
        # data_test_processed[target_vars[-1]] = durations_test
        # data_test_processed[target_vars[-1]] = data_test_processed[target_vars[-1]].astype(int) # Convertsurvival times to integers

            #########################################################
        
    elif additional_dr_options['model'] == 'none':
        print("NO FURTHER DIMENSIONALITY REDUCTION PERFORMED")
        data_train_processed = data_train
        data_test_processed = data_test
        encoder = None
    
    return(encoder, data_train_processed, data_test_processed)
  
# def setup_autoencoder(X, y, HIDDEN_DIM, LATENT_SPACE):
#     """
#     Creates a tensorflow (keras) encoder/decoder model and returns the
#     parts of the model used later during training and to pull the latent
#     features.
#     """    
#     import tensorflow as tf
#     from tensorflow.keras.layers import Input, Dense
#     from tensorflow.keras.models import Model 

#     input_dim = X.shape[1]
#     output_dim = y.shape[1]
#     print(f'Input dimensions {input_dim} - output dimensions {output_dim}')
    
#     original_inputs = Input(shape=(input_dim,))

#     # encoder
#     x = tf.keras.layers.Dense(HIDDEN_DIM, activation=None, kernel_initializer='random_uniform')(original_inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(LATENT_SPACE, activation=None, kernel_initializer='random_uniform')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
#     enc_out = tf.keras.layers.Dropout(0.2)(x)

#     # decoder
#     x_d = tf.keras.layers.Dense(HIDDEN_DIM, activation=None, kernel_initializer='random_uniform')(enc_out)
#     x_d = tf.keras.layers.BatchNormalization()(x_d)
#     x_d = tf.keras.layers.ReLU()(x_d)
#     x_d = tf.keras.layers.Dropout(0.2)(x_d)
#     x_d = tf.keras.layers.Dense(output_dim, activation=None, kernel_initializer='random_uniform')(x_d)
#     x_d = tf.keras.layers.BatchNormalization()(x_d)
#     dec_out = tf.keras.layers.ReLU()(x_d)
    
#     return original_inputs, enc_out, dec_out


# def train_autoencoder(data_, FRAC_TRAIN_AE, HIDDEN_DIM, LATENT_SPACE, EPOCHS, BATCH_SIZE, ES_PATIENCE):
#     """
#     Trains Encoder/Decoder Models
    
#     Returns encoder, decoder & training history
#     """
#     import tensorflow as tf
#     from tensorflow.keras.layers import Input, Dense
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.callbacks import EarlyStopping
#     import matplotlib.pylab as plt

#     X_train = data_
#     Y_train = X_train #For now vanilla AE
    
#     #Create Model
#     original_inputs, enc_out, dec_out = setup_autoencoder(X_train, Y_train, HIDDEN_DIM, LATENT_SPACE)
#     autoencoder = Model(inputs=original_inputs, outputs=dec_out)

#     #Optimization hyperparameters: Can refine
#     loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01)
#     opt = tf.keras.optimizers.Adam(lr=0.005)
#     autoencoder.compile(optimizer=opt, loss=loss_fn)

#     #Callbacks (early stopping):
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=ES_PATIENCE, restore_best_weights=True)

#     autoencoder.fit(X_train, Y_train,
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     verbose=1,
#                     shuffle=True,
#                     callbacks = [es],
#                     validation_split=1-FRAC_TRAIN_AE)

#     history_df = pd.DataFrame(autoencoder.history.history)
#     history_df[['loss','val_loss']] \
#         .plot(figsize=(10, 5),
#              title='Train/Val Loss')
#     plt.tight_layout()
#     plt.savefig(f'./AE_loss.png')

#     # create encoder model
#     encoder = Model(inputs=original_inputs, outputs=enc_out)
#     # create decoder model
#     encoded_input = Input(shape=(LATENT_SPACE,))
#     decoder_part = tf.keras.Sequential(autoencoder.layers[-7:])#hard coded; see if this needs to change
#     decoder = Model(inputs=encoded_input, outputs=decoder_part(encoded_input))
    
#     return encoder, decoder, history_df

# def encode(data, encoder):
#     """
#     Uses the encoder of the AE to create latent features
#     from the raw input data (X)
#     """
#     X = data.copy() # Work on a copy of the input data
    
#     latent_vector = encoder.predict(X)
    
#     #Will only retain the latent features:
#     cols_to_retain = []
#     for i in range(latent_vector.shape[1]):
#         data[f'latent_{i}'] = latent_vector[:,i]
#         cols_to_retain.append(f'latent_{i}')
    
#     #Remove original features
#     data = data[cols_to_retain]

#     return data 