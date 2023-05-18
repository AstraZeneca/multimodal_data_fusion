import os, sys, glob
import numpy as np
import pandas as pd
import scipy.stats as st
import argparse
import random
import pickle
from modeling.feature_modelling import feature_selection_pipeline, get_features_per_modality
from modeling.survival_modelling import survival_modelling_main, get_weighted_ensemble_predictions
from modeling.late_fusion_modelling import late_fusion
from configs.pipeline_config import pipeline_config
from configs.base_config import path_config
from utils.data_utils import load_data, split_data
from utils.config_utils import save_config


def getModels(num_runs: int, survival_models: list, data_train: pd.DataFrame, data_val: pd.DataFrame, data_test: pd.DataFrame, targets: list, debug: int, return_feature_importance_list: int) -> pd.DataFrame:
    trained_models = pd.DataFrame(
        None, index=range(num_runs), columns=survival_models)
    for model in survival_models:
        # NOTE: Ask Nikos about adding data_val here
        surv_model, test_set_c_index, val_set_c_index, train_set_c_index, importances_df, pred_train, pred_test = survival_modelling_main(
            data_train, data_test, targets, model, return_feature_importance_list, debug)

        trained_models.at[trained_models.index[run], model] = surv_model
        return trained_models


def getDataPerModality(subset: list, modality: str, data_train_all_modalities: pd.DataFrame, data_val_all_modalities: pd.DataFrame, data_test_all_modalities: pd.DataFrame, features_per_modality: dict, debug: int, time_col: str, evt_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_train = data_train_all_modalities[features_per_modality[subset[modality]]]
    data_val = data_val_all_modalities[features_per_modality[subset[modality]]]
    data_test = data_test_all_modalities[features_per_modality[subset[modality]]]
    if debug == 1:
        print("Dimensions of "+subset[modality] +
              " training dataset: "+str(data_train.shape))
        print("Dimensions of "+subset[modality] +
              " validation dataset: "+str(data_train.shape))
        print("Dimensions of "+subset[modality] +
              " test dataset: "+str(data_test.shape))

    #Add TARGETS & CANCER_TYPE-if needed
    data_train[time_col] = data_train_all_modalities[time_col]
    data_train[evt_col] = data_train_all_modalities[evt_col]
    #data_train['SUBJID'] = SUBJID_TR#data_train_all_modalities['bcr_patient_barcode']
    data_val[time_col] = data_val_all_modalities[time_col]
    data_val[evt_col] = data_val_all_modalities[evt_col]
    #data_val['SUBJID'] = SUBJID_VAL#data_val_all_modalities['bcr_patient_barcode']
    data_test[time_col] = data_test_all_modalities[time_col]
    data_test[evt_col] = data_test_all_modalities[evt_col]
    #data_test['SUBJID'] = SUBJID_TE#data_test_all_modalities['bcr_patient_barcode']
    if 'cancer_type' in list(data_train_all_modalities.columns):
        data_train['cancer_type'] = data_train_all_modalities['cancer_type']
        data_val['cancer_type'] = data_val_all_modalities['cancer_type']
        data_test['cancer_type'] = data_test_all_modalities['cancer_type']
    return data_train, data_val, data_test


def pipeline(args):
    if args.expt is None:
        expt_name = 'default'
    else:
        expt_name = '{:06}'.format(args.expt)
    expt_config, exptname = pipeline_config(expt_name)
    datapath = args.csv

    # setup my custom paths
    mypaths, myusername = path_config()
    # Savepath settings
    save_path = os.path.join(mypaths['output'], 'Modeling-expt-'+expt_name)
    load_path = save_path  # variable initialization
    # Archival settings
    load_from_previous = not expt_config['ArchivalSettings']['Enabled']
    if load_from_previous:
        load_path = os.path.join(mypaths['output'], 'Modeling-expt-'+expt_config['ArchivalSettings']['WhenDisabledLoadFromExpt'])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    output_name = 'Modeling-expt-'+expt_name

    # Modelling settings
    num_runs = 1
    if expt_config['modelling_settings']['sensitivity_analysis']:
        num_runs = expt_config['modelling_settings']['num_runs']  # Number of full runs of entire pipeline on a different train/test split
    subset_list = expt_config['modelling_settings']['subsets']  # Modality subset
    survival_models = expt_config['modelling_settings']['models']
    evaluate_on_individual_types = expt_config['modelling_settings']['evaluate_on_all_cancers']
    
    # Data Settings
    os_col = expt_config['modelling_settings']['data_characteristics']['targets'][1]
    evt_col = expt_config['modelling_settings']['data_characteristics']['targets'][0]
    #Will store C-index of base models for each run:
    all_runs_test = pd.DataFrame(np.nan, index=range(num_runs), columns=survival_models)
    all_runs_val = pd.DataFrame(np.nan, index=range(num_runs), columns=survival_models)
    all_runs_train = pd.DataFrame(np.nan, index=range(num_runs), columns=survival_models)

    # Will store survival models trained on each run:
    trained_models_per_modality = {}

    #Will store Test set C-index of weighted ensemble for each run:
    full_train_weighted_ensemble_c_index = np.zeros((num_runs, 1))
    train_weighted_ensemble_c_index = np.zeros((num_runs, 1))

    # User options
    user_options_settings = expt_config['user_options_settings']


    print("\n   #########################")

    # Load the data
    data = load_data(datapath)
    cancer_types = expt_config['data_settings']['cancer_types']
    # Model for each subset of cancer types
    for cancer_type in cancer_types:
        if evaluate_on_individual_types == 1 and len(cancer_type) > 1:

            # Will store per-cancer results:
            all_runs_test_type = {}
            all_runs_train_type = {}

            # Will store per-cancer importances:
            importances_full_df_type = {}

            # Will store C-index of base models for each run:
            for indiv_type in cancer_type:
                all_runs_test_type[indiv_type] = pd.DataFrame(
                    np.nan, index=range(num_runs), columns=survival_models)
                all_runs_train_type[indiv_type] = pd.DataFrame(np.nan, index=range(num_runs), columns=survival_models)
            
            #Will store Test set C-index of weighted ensemble for each run per cancer type:
            full_train_weighted_ensemble_c_index_type = np.zeros((num_runs, len(cancer_type)))
            train_weighted_ensemble_c_index_type = np.zeros((num_runs, len(cancer_type)))
            
            #Will store ensemble weights for each run:
            ensemble_weights_tr = np.zeros((num_runs, len(survival_models)))
            
            #Will store different ensemble weights for each run, per cancer type:
            ensemble_weights_tr_type = np.zeros((num_runs, len(cancer_type), len(survival_models)))
        
        for subset in subset_list:
            subset_folder = os.path.join(save_path, '_'.join(subset))
            load_subset_folder = os.path.join(load_path, '_'.join(subset))
            print("\n Cancer Types(s):", cancer_type)
            print('\n Modalities: ', subset)
            print("\n Starting {} runs ... \n".format(num_runs))

            #Will store dataset splits (upon preprocessing & feature reduction) obtained on each run:
            datasets_train_per_run_per_modality = pd.DataFrame(None, index=range(num_runs), columns=subset)
            datasets_val_per_run_per_modality = pd.DataFrame(None, index=range(num_runs), columns=subset)
            datasets_test_per_run_per_modality = pd.DataFrame(None, index=range(num_runs), columns=subset)

            # Store feature importances for the given subset:
            importances_full_df = pd.DataFrame(columns=['Feature', 'Feature Importance', 'Feature Importance STD',
                                            'Model', 'Test C-index', 'Validation C-index', 'Train C-index', 'Rank'])
            importances_full_df_type = pd.DataFrame(
                columns=['Feature', 'Feature Importance', 'Feature Importance STD', 'Model', 'Cancer Type' 'Test C-index', 'Train C-index', 'Rank'])
            selected_features = {}
            # Add seeds for random splits of subject ids or rows.
            random_splits = []
            if load_from_previous:
                random_splits = np.load(os.path.join(load_subset_folder, 'split_indices.npy'))
            else:
                # If continuing execution ...
                if os.path.exists(os.path.join(subset_folder, 'split_indices.npy')):
                    random_splits = np.load(os.path.join(subset_folder, 'split_indices.npy'))
                else:
                    if not os.path.isdir(subset_folder):
                        os.makedirs(subset_folder)
                    random_splits = random.sample(range(1, 1000), num_runs)
                    np.save(os.path.join(subset_folder, 'split_indices.npy'), random_splits)
            # Will store survival models trained on each run:
            trained_models = pd.DataFrame(None, index=range(num_runs), columns=survival_models)
            for run in range(num_runs):
                pred_true_train = pd.DataFrame(columns=['True OS', 'CNSR', 'Cancer Type'])
                pred_true_val = pd.DataFrame(columns=['True OS', 'CNSR', 'Cancer Type'])
                pred_true_test = pd.DataFrame(columns=['True OS', 'CNSR', 'Cancer Type'])
                print('*******Initiating run: '+str(run+1)+'/'+str(num_runs))
                # Read the cohort here based on config param and use if needed style
                if not expt_config['modelling_settings']['sensitivity_analysis'] and expt_config['modelling_settings']['cohorts_settings']['UseCohorts']:
                    if expt_config['modelling_settings']['cohorts_settings']['CohortsAvailable']:
                        cohort_col = expt_config['modelling_settings']['cohorts_settings']['CohortColumn']
                        cohort_attr = expt_config['modelling_settings']['cohorts_settings']['CohortAttributes']
                        cohort_path = expt_config['modelling_settings']['cohorts_settings']['CohortTable']
                        cohorts = pd.read_csv(cohort_path)
                        dataframes = []
                        for label in expt_config['modelling_settings']['cohorts_settings']['CohortSplits']:
                            dataframes.append(split_data(data=data, cohorts=cohorts[cohorts[cohort_col]==label], attr=cohort_attr))
                        # if len(expt_config['modelling_settings']['cohorts_settings']['CohortSplits']) == 3:
                        #     # Combine training and validation
                        #     dataframes[0] = pd.concat([dataframes[0], dataframes[1]])
                        #     del dataframes[1]
                    else:
                        raise ValueError("Cohorts Unavailable for analysis. Please check config settings")
                else:
                    dataframes = [data]
                if not load_from_previous:
                    if os.path.exists(os.path.join(subset_folder, str(run+1) + '_data.pkl')):
                        run_file = open(os.path.join(subset_folder, str(run+1) + '_data.pkl'), "rb")
                        data_train = pickle.load(run_file)
                        data_val = pickle.load(run_file)
                        data_test = pickle.load(run_file)
                        run_file.close()
                    else:
                        #--------- Perform subset selection, train-test split, preprocessing & feature selection: --------
                        data_train, data_val, data_test = feature_selection_pipeline(data=dataframes, modalities_dict=expt_config['settings']['modalities'], subset=subset,
                                                                        random_split=random_splits[run], include_PDL1_status=expt_config['modelling_settings']['pdl1'],
                                                                        modality_handling_mode=expt_config['feature_selection_settings']['modality_handling'], 
                                                                        subset_selection_mode=expt_config['feature_selection_settings']['subset_selection_mode'], 
                                                                        preprocess_modality_dict=expt_config['settings']['preprocessing'], 
                                                                        frac_train=expt_config['settings']['train_fraction'], split_mode=expt_config['settings']['split_mode'],
                                                                        data_char_dict=expt_config['modelling_settings']['data_characteristics'],
                                                                        unsupervised_fs_options_dict=expt_config['feature_selection_settings']['unsupervised'], 
                                                                        supervised_fs_options_dict=expt_config['feature_selection_settings']['supervised'], 
                                                                        fs_consistency_options_dict=expt_config['feature_selection_settings']['consistency_options'], 
                                                                        user_options_dict=user_options_settings, stratify=expt_config['settings']['stratify'],
                                                                        strat_cols=expt_config['settings']['strat_columns'])
                    run_file = open(os.path.join(subset_folder, str(run+1) + '_data.pkl'), "wb")
                    pickle.dump(data_train, run_file)
                    pickle.dump(data_val, run_file)
                    pickle.dump(data_test, run_file)
                    run_file.close()
                    #------------------------------------------------------------------------------------------------
                    selected_features['run'+str(run+1)] = data_train.columns[:-2].tolist()
                else:
                    if os.path.exists(os.path.join(load_subset_folder, str(run+1) + '_data.pkl')):
                        run_file = open(os.path.join(load_subset_folder, str(run+1) + '_data.pkl'), "rb")
                        data_train = pickle.load(run_file)
                        data_val = pickle.load(run_file)
                        data_test = pickle.load(run_file)
                        run_file.close()
                
                #-------------------Steps below should be applied on each modality separately--------------------
                data_train_all_modalities = data_train.copy()
                data_val_all_modalities = data_val.copy()
                data_test_all_modalities = data_test.copy()
                
                features_per_modality = get_features_per_modality(data_train, expt_config['settings']['modalities'])
                # NOTE: Ask Nikos if this is where the code changes to be for late fusion
                if expt_config['modelling_settings']['late_fusion']:
                    for modality in range(len(subset)):
                        data_train, data_val, data_test = getDataPerModality(
                            subset, modality, data_train_all_modalities, data_val_all_modalities, data_test_all_modalities, features_per_modality, user_options_settings['verbose'], os_col, evt_col)

                        # Save data_train & data_test & for curent modality & current run
                        datasets_train_per_run_per_modality.at[all_runs_train.index[run],
                                                               subset[modality]] = data_train
                        datasets_val_per_run_per_modality.at[all_runs_val.index[run],
                                                             subset[modality]] = data_val
                        datasets_test_per_run_per_modality.at[all_runs_test.index[run],
                                                              subset[modality]] = data_test
                        #------------------------------------- Survival Modelling: --------------------------------------
                        pred_true_train['True OS'] = data_train[os_col]
                        pred_true_val['True OS'] = data_val[os_col]
                        pred_true_test['True OS'] = data_test[os_col]
                        pred_true_train['CNSR'] = data_train[evt_col]
                        pred_true_val['CNSR'] = data_val[evt_col]
                        pred_true_test['CNSR'] = data_test[evt_col]
                        # Running all the Selected Survival Models    
                        trained_models = getModels(num_runs, survival_models, data_train, data_val, data_test, expt_config['modelling_settings'][
                                                'data_characteristics']['targets'], user_options_settings['verbose'], user_options_settings['return_feature_importance_list'])
                else:
                    pred_true_train['True OS'] = data_train[os_col]
                    pred_true_val['True OS'] = data_val[os_col]
                    pred_true_test['True OS'] = data_test[os_col]
                    pred_true_train['CNSR'] = data_train[evt_col]
                    pred_true_val['CNSR'] = data_val[evt_col]
                    pred_true_test['CNSR'] = data_test[evt_col]
                    trained_models = getModels(num_runs, survival_models, data_train, data_val, data_test, expt_config['modelling_settings'][
                                                'data_characteristics']['targets'], user_options_settings['verbose'], user_options_settings['return_feature_importance_list'])
                #Store models & their training / validation set performances to compute ensemble predictions:
                training_c_indices_run = all_runs_train.iloc[run].to_numpy() #all_runs_val.iloc[run].to_numpy() #Choose training or validation set weights
                model_pool_run = trained_models.iloc[run]
                full_train_weighted_ensemble_c_index[run], train_weighted_ensemble_c_index[run] = get_weighted_ensemble_predictions(data_train, data_test, expt_config['modelling_settings']['data_characteristics']['targets'], survival_models, model_pool_run, training_c_indices_run, expt_config['user_options_settings']['verbose'])    
                #all_runs_test.at[all_runs_train.index[run], 'Full Train Weighted Ensemble'] = full_train_weighted_ensemble_c_index[run] #ignore this; we almost never get a model with <0.5 training set C-index
                all_runs_test.at[all_runs_train.index[run], 'Train Weighted Ensemble'] = train_weighted_ensemble_c_index[run]
                # else:
                #     # TODO: Load the all_runs files and importances
                #     if os.path.exists(os.path.join(load_subset_folder, str(run+1) + '.pkl')):
                #         run_file = open(os.path.join(load_subset_folder, str(run+1) + '.pkl'), "rb")
                #         data_train = pickle.load(run_file)
                pred_true_train.to_csv(os.path.join(save_path,'_'.join(subset)+'_RUN' + str(run+1)+'_PRED_TRUE_TRAIN.csv'))
                pred_true_test.to_csv(os.path.join(save_path,'_'.join(subset)+'_RUN' + str(run+1)+'_PRED_TRUE_TEST.csv'))
            ### Save config ###
            save_config(expt_config=expt_config, expt_name=output_name, savepath=save_path)
        
        #------------------------------------------------------------------------------------------------------    
        # Save list of selected features in each run
        if not load_from_previous:
            df_feats = pd.DataFrame.from_dict(selected_features)
            df_feats.to_csv(os.path.join(subset_folder, 'selected_features.csv'))
            # Save model performances
            all_runs_train.to_csv(os.path.join(save_path, 'modelPerformance_train.csv'))
            all_runs_val.to_csv(os.path.join(save_path, 'modelPerformance_val.csv'))
            all_runs_test.to_csv(os.path.join(save_path, 'modelPerformance_test.csv'))

        #------------------------------------- Printing Results: ----------------------------------------------
        # Computing model statistics over all runs:
        print("\n   Finished {} runs ... \n".format(num_runs))
        print("   Modalities: ", subset)

        print('*************************FINAL RESULTS**************************')
        print("   Test set C-index per model, per run, on all data:")
        print(all_runs_test)

        for model in all_runs_test.columns:
            print('Model ' + model)
            model_mean = all_runs_test[model].mean(axis=0, skipna = True)#Model mean, across runs
            sc = st.sem(all_runs_test[model], nan_policy = 'omit')#Model standard error around the mean, across runs
            print("   Runs mean: {}, CI: {}".format(model_mean, st.t.interval(alpha=0.95, df=num_runs - 1, loc=model_mean, scale=sc)))

        #Show top num_feat feature importances:
        num_feat = 20
        if importances_full_df.shape[0] == 0:
            print('Feature importance list is empty.')
        else:
            print('Feature Importances (top-'+str(num_feat)+') -- on all test data:')
            average_importance = importances_full_df.groupby('Feature')['Feature Importance'].mean().sort_values(ascending=False)
            # Save results
            average_importance.to_csv(os.path.join(save_path, 'Average_importances.csv'))
            print(average_importance[:num_feat])
        print('******************************************************************') 
        #------------------------------------------------------------------------------------------------------
        print("\n   @@@@@@@@@@@@@@@@@@@@@@\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Survival Pipeline")
    parser.add_argument('--expt', '-e', default=None, type=int, help='Experiment version to be run')
    parser.add_argument('--csv', '-c', default='clinical_radiomics_features.csv', 
                        type=str, help='Full path to the features csv')
    args = parser.parse_args()
    pipeline(args)
