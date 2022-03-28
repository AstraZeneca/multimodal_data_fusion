import os, sys, glob
import numpy as np
import pandas as pd
import scipy.stats as st
import argparse
import random
import pickle
from RaDx_modelling import feature_selection_pipeline
from survival_modelling import survival_modelling_main, get_weighted_ensemble_predictions
from configs.pipeline_config import pipeline_config
from configs.base_config import path_config
from utils.data_utils import load_data, split_data
from utils.config_utils import save_config


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

    #Will store C-index of base models for each run:
    all_runs_test = pd.DataFrame(np.nan, index=range(num_runs), columns=survival_models)
    all_runs_val = pd.DataFrame(np.nan, index=range(num_runs), columns=survival_models)
    all_runs_train = pd.DataFrame(np.nan, index=range(num_runs), columns=survival_models)

    #Will store survival models trained on each run:
    trained_models = pd.DataFrame(None, index=range(num_runs), columns=survival_models)

    #Will store Test set C-index of weighted ensemble for each run:
    full_train_weighted_ensemble_c_index = np.zeros((num_runs, 1))
    train_weighted_ensemble_c_index = np.zeros((num_runs, 1))

    # User options
    user_options_settings = expt_config['user_options_settings']


    print("\n   #########################")

    # Load the data
    data = load_data(datapath)

    for subset in subset_list:
        subset_folder = os.path.join(save_path, '_'.join(subset))
        load_subset_folder = os.path.join(load_path, '_'.join(subset))
        print('   Modalities: ', subset)
        print("   Starting {} runs ... \n".format(num_runs))

        #Store feature importances for the given subset:
        importances_full_df = pd.DataFrame(columns=['Feature', 'Feature Importance', 'Feature Importance STD', 'Model', 'Test C-index', 'Validation C-index', 'Train C-index', 'Rank'])

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
        
        for run in range(num_runs):
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
                    if len(expt_config['modelling_settings']['cohorts_settings']['CohortSplits']) == 3:
                        # Combine training and validation
                        dataframes[0] = pd.concat([dataframes[0], dataframes[1]])
                        del dataframes[1]
                else:
                    raise ValueError("Cohorts Unavailable for analysis. Please check config settings")
            else:
                dataframes = [data]
            if not load_from_previous:
                if os.path.exists(os.path.join(subset_folder, str(run+1) + '_data.pkl')):
                    run_file = open(os.path.join(subset_folder, str(run+1) + '_data.pkl'), "rb")
                    data_train = pickle.load(run_file) 
                    data_test = pickle.load(run_file)
                    run_file.close()
                else:
                    #--------- Perform subset selection, train-test split, preprocessing & feature selection: --------
                    data_train, data_test = feature_selection_pipeline(data=dataframes, modalities_dict=expt_config['settings']['modalities'], subset=subset,
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
                pickle.dump(data_test, run_file)
                run_file.close()
                #------------------------------------------------------------------------------------------------
                selected_features['run'+str(run+1)] = data_train.columns[:-2].tolist()
            else:
                if os.path.exists(os.path.join(load_subset_folder, str(run+1) + '_data.pkl')):
                    run_file = open(os.path.join(load_subset_folder, str(run+1) + '_data.pkl'), "rb")
                    data_train = pickle.load(run_file) 
                    data_test = pickle.load(run_file)
                    run_file.close()
            #------------------------------------- Survival Modelling: --------------------------------------
            # TODO: Save prediction for each subject id and all train/test split subject ids
            # Running all the Selected Survival Models    
            for model in survival_models:
                surv_model, test_set_c_index, val_set_c_index, train_set_c_index, importances_df = survival_modelling_main(data_train, data_test, model, user_options_settings['return_feature_importance_list'], user_options_settings['verbose'])

                # TODO: Save the model and data point prediction
            
                #Store test set, val set & train_set C-index attained by model:
                all_runs_test.at[all_runs_test.index[run], model] = test_set_c_index
                all_runs_val.at[all_runs_val.index[run], model] = val_set_c_index
                all_runs_train.at[all_runs_train.index[run], model] = train_set_c_index
                importances_df['Run'] = run
                trained_models.at[all_runs_val.index[run], model] = surv_model

                #Concatenate feature importances of all models:
                importances_full_df = pd.concat([importances_df, importances_full_df])

            #Store models & their training / validation set performances to compute ensemble predictions:
            training_c_indices_run = all_runs_train.iloc[run].to_numpy() #all_runs_val.iloc[run].to_numpy() #Choose training or validation set weights
            model_pool_run = trained_models.iloc[run]
            full_train_weighted_ensemble_c_index[run], train_weighted_ensemble_c_index[run] = get_weighted_ensemble_predictions(data_train, data_test, survival_models, model_pool_run, training_c_indices_run, expt_config['user_options_settings']['verbose'])    
            #all_runs_test.at[all_runs_train.index[run], 'Full Train Weighted Ensemble'] = full_train_weighted_ensemble_c_index[run] #ignore this; we almost never get a model with <0.5 training set C-index
            all_runs_test.at[all_runs_train.index[run], 'Train Weighted Ensemble'] = train_weighted_ensemble_c_index[run]
            # else:
            #     # TODO: Load the all_runs files and importances
            #     if os.path.exists(os.path.join(load_subset_folder, str(run+1) + '.pkl')):
            #         run_file = open(os.path.join(load_subset_folder, str(run+1) + '.pkl'), "rb")
            #         data_train = pickle.load(run_file)

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
    main(args)
