from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
import scipy.stats as st
from typing import Tuple


def late_fusion(experiment_name: str, n_runs: int, cancer_type: list, modality_list: list, model_list: list, c_index_threshold_to_include: float = 0.53) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_results = {}
    val_results = {}
    test_results = {}
    # Part 1 -- Find Weights on validation set
    for i in range(n_runs):
        train_results['Run'+str(i+1)] = pd.read_csv(experiment_name+'_RUN'+str(i+1)+'_PRED_TRUE_TRAIN.csv')
        val_results['Run'+str(i+1)] = pd.read_csv(experiment_name +'_RUN'+str(i+1)+'_PRED_TRUE_VALIDATION.csv')
        test_results['Run'+str(i+1)] = pd.read_csv(experiment_name+'_RUN'+str(i+1)+'_PRED_TRUE_TEST.csv')

        # Check this
        if cancer_type != 'ALL':
            train_results['Run'+str(i+1)] = train_results['Run'+str(i+1)].drop(train_results['Run'+str(
                i+1)][train_results['Run'+str(i+1)]['Cancer Type'] != cancer_type].index)
            val_results['Run'+str(i+1)] = val_results['Run'+str(i+1)].drop(val_results['Run'+str(
                i+1)][val_results['Run'+str(i+1)]['Cancer Type'] != cancer_type].index)
            test_results['Run'+str(i+1)] = test_results['Run'+str(i+1)].drop(test_results['Run'+str(
                i+1)][test_results['Run'+str(i+1)]['Cancer Type'] != cancer_type].index)

        # use val in place of training for purposes of getting weights
        train_results['Run'+str(i+1)] = val_results['Run'+str(i+1)]

    # Find c-index of each model and normalize their predictions; these will be used for the ensemble prediction
    # and for the fused (late fusion multimodal) prediction:
    c_indices_train = {}
    c_indices_test = {}

    predictions_train_normalized = {}
    predictions_test_normalized = {}

    for i in range(n_runs):
        c_indices_train['Run'+str(i+1)] = {}
        c_indices_test['Run'+str(i+1)] = {}
        predictions_train_normalized['Run'+str(i+1)] = {}
        predictions_test_normalized['Run'+str(i+1)] = {}
        for model in model_list:
            for modality in modality_list:
                c_indices_train['Run'+str(i+1)][model+'_'+modality] = concordance_index(train_results['Run'+str(i+1)]['True OS'], -
                                                                                        train_results['Run'+str(i+1)]['Predicted Risk Score '+model+'_'+modality], train_results['Run'+str(i+1)]['CNSR'])
                c_indices_test['Run'+str(i+1)][model+'_'+modality] = concordance_index(test_results['Run'+str(i+1)]['True OS'], -
                                                                                       test_results['Run'+str(i+1)]['Predicted Risk Score '+model+'_'+modality], test_results['Run'+str(i+1)]['CNSR'])

                # Get normalized predictions per model, per modality, per run:
                predictions_train = train_results['Run'+str(
                    i+1)]['Predicted Risk Score '+model+'_'+modality]
                predictions_test = test_results['Run' +
                                                str(i+1)]['Predicted Risk Score '+model+'_'+modality]
                if predictions_train.max()-predictions_train.min() != 0:
                    predictions_train_normalized['Run'+str(i+1)][model+'_'+modality] = (
                        predictions_train-predictions_train.min())/(predictions_train.max()-predictions_train.min())
                # if all training set predictions of this model on that modality have the same value, set all normalized predictions to 0.5 (fully uncertain)
                else:
                    predictions_train_normalized['Run'+str(
                        i+1)][model+'_'+modality] = 0.5*np.ones(predictions_train.shape)
                if predictions_test.max()-predictions_test.min() != 0:
                    predictions_test_normalized['Run'+str(i+1)][model+'_'+modality] = (
                        predictions_test-predictions_test.min())/(predictions_test.max()-predictions_test.min())
                # if all test set predictions of this model on that modality have the same value, set all normalized predictions to 0.5 (fully uncertain)
                else:
                    predictions_test_normalized['Run'+str(
                        i+1)][model+'_'+modality] = 0.5*np.ones(predictions_test.shape)

    # Get model weights for ensemble (per run, per modality):
    model_weights = {}

    for i in range(n_runs):
        model_weights['Run'+str(i+1)] = {}
        for modality in modality_list:
            c_indices_train_array = []
            for model in model_list:
                c_indices_train_array.append(
                    c_indices_train['Run'+str(i+1)][model+'_'+modality])

            c_indices_train_array = np.array(c_indices_train_array)
            model_weights_ = c_indices_train_array - \
                c_index_threshold_to_include * \
                np.ones((c_indices_train_array.shape))
            # models with C-index < 0.5 are assigned a weight of 0
            model_weights_[model_weights_ < 0] = 0
            model_weights_ = model_weights_/np.sum(model_weights_)

            # If all weights are 0, then normalization above makes them nan; replace with 0
            if np.isnan(model_weights_).all():
                model_weights_ = np.zeros(np.shape(model_weights_))

            model_weights['Run'+str(i+1)][modality] = model_weights_

    # FUSE MODALITIES per each model:

    # Get modality weights for modality fusion (per run, per model):
    modality_weights = {}

    for i in range(n_runs):
        modality_weights['Run'+str(i+1)] = {}
        for model in model_list:
            c_indices_train_array = []
            for modality in modality_list:
                c_indices_train_array.append(
                    c_indices_train['Run'+str(i+1)][model+'_'+modality])

            c_indices_train_array = np.array(c_indices_train_array)
            modality_weights_ = c_indices_train_array - \
                c_index_threshold_to_include * \
                np.ones((c_indices_train_array.shape))
            # models with C-index < 0.5 are assigned a weight of 0
            modality_weights_[modality_weights_ < 0] = 0
            modality_weights_ = modality_weights_/np.sum(modality_weights_)

            # If all weights are 0, then normalization above makes them nan; replace with 0
            if np.isnan(modality_weights_).all():
                modality_weights_ = np.zeros(np.shape(modality_weights_))

            # modality_weights_ = np.ones(c_indices_train_array.shape)*(1/len(modality_list)) #Equal weights per modality

            modality_weights['Run'+str(i+1)][model] = modality_weights_

    # Part 2 use models trained on full training + validation set
    # NOTE: Ask Nikos about the model trained on combo. Where is this?
    train_results = {}
    test_results = {}

    for i in range(n_runs):
        train_results['Run'+str(i+1)] = pd.read_csv(experiment_name +
                                                    '_RUN'+str(i+1)+'_PRED_TRUE_TRAIN.csv')
        test_results['Run'+str(i+1)] = pd.read_csv(experiment_name +
                                                   '_RUN'+str(i+1)+'_PRED_TRUE_TEST.csv')

        # filter by type?
        if cancer_type != 'ALL':
            train_results['Run'+str(i+1)] = train_results['Run'+str(i+1)].drop(train_results['Run'+str(
                i+1)][train_results['Run'+str(i+1)]['Cancer Type'] != cancer_type].index)
            test_results['Run'+str(i+1)] = test_results['Run'+str(i+1)].drop(test_results['Run'+str(
                i+1)][test_results['Run'+str(i+1)]['Cancer Type'] != cancer_type].index)

    # Find c-index of each model and normalize their predictions; these will be used for the ensemble prediction
    # and for the fused (late fusion multimodal) prediction:
    c_indices_train = {}
    c_indices_test = {}

    predictions_train_normalized = {}
    predictions_test_normalized = {}

    for i in range(n_runs):
        c_indices_train['Run'+str(i+1)] = {}
        c_indices_test['Run'+str(i+1)] = {}
        predictions_train_normalized['Run'+str(i+1)] = {}
        predictions_test_normalized['Run'+str(i+1)] = {}
        for model in model_list:
            for modality in modality_list:
                c_indices_train['Run'+str(i+1)][model+'_'+modality] = concordance_index(train_results['Run'+str(i+1)]['True OS'], -
                                                                                        train_results['Run'+str(i+1)]['Predicted Risk Score '+model+'_'+modality], train_results['Run'+str(i+1)]['CNSR'])
                c_indices_test['Run'+str(i+1)][model+'_'+modality] = concordance_index(test_results['Run'+str(i+1)]['True OS'], -
                                                                                       test_results['Run'+str(i+1)]['Predicted Risk Score '+model+'_'+modality], test_results['Run'+str(i+1)]['CNSR'])

                # Get normalized predictions per model, per modality, per run:
                predictions_train = train_results['Run'+str(
                    i+1)]['Predicted Risk Score '+model+'_'+modality]
                predictions_test = test_results['Run' +
                                                str(i+1)]['Predicted Risk Score '+model+'_'+modality]
                if predictions_train.max()-predictions_train.min() != 0:
                    predictions_train_normalized['Run'+str(i+1)][model+'_'+modality] = (
                        predictions_train-predictions_train.min())/(predictions_train.max()-predictions_train.min())
                # if all training set predictions of this model on that modality have the same value, set all normalized predictions to 0.5 (fully uncertain)
                else:
                    predictions_train_normalized['Run'+str(
                        i+1)][model+'_'+modality] = 0.5*np.ones(predictions_train.shape)
                if predictions_test.max()-predictions_test.min() != 0:
                    predictions_test_normalized['Run'+str(i+1)][model+'_'+modality] = (
                        predictions_test-predictions_test.min())/(predictions_test.max()-predictions_test.min())
                # if all test set predictions of this model on that modality have the same value, set all normalized predictions to 0.5 (fully uncertain)
                else:
                    predictions_test_normalized['Run'+str(
                        i+1)][model+'_'+modality] = 0.5*np.ones(predictions_test.shape)

    # Final prediction of fused models -- TODO: Add ensemble with learned weights? # First, add training set predictions?
    fused_prediction = {}
    modality_weight_list = {}
    for i in range(n_runs):
        fused_prediction['Run'+str(i+1)] = {}
        modality_weight_list['Run'+str(i+1)] = {}
        num_test_datapoints = len(test_results['Run'+str(i+1)])
        for model in model_list:
            predictions_normalized_p_modality = np.zeros((num_test_datapoints, len(
                modality_list)))  # [] #store  only the predictions for that modality
            for modality_i in range(len(modality_list)):
                # .append(np.array(predictions_train_normalized['Run'+str(i+1)][model+'_'+modality]))
                predictions_normalized_p_modality[:, modality_i] = predictions_test_normalized['Run'+str(
                    i+1)][model+'_'+modality_list[modality_i]]

            weights = np.transpose(np.expand_dims(
                modality_weights['Run'+str(i+1)][model], axis=-1))
            modality_weight_list['Run'+str(i+1)][model] = weights
            fused_prediction['Run'+str(i+1)][model] = np.multiply(
                weights, predictions_normalized_p_modality).sum(axis=1)
            predictions_test_normalized['Run'+str(
                i+1)][model+'_FUSED'] = fused_prediction['Run'+str(i+1)][model]

    for i in range(n_runs):
        for model in model_list:
            c_indices_test['Run'+str(i+1)][model+'_FUSED'] = concordance_index(test_results['Run'+str(
                i+1)]['True OS'], -fused_prediction['Run'+str(i+1)][model], test_results['Run'+str(i+1)]['CNSR'])

    # Extend modalities to include fused modality:
    modality_list.append('FUSED')

    # Final prediction of ensemble -- TODO: Add ensemble with learned weights? # First, add training set predictions?
    ensemble_prediction = {}
    #ensemble_w_flip_prediction = {}
    for i in range(n_runs):
        ensemble_prediction['Run'+str(i+1)] = {}
        #ensemble_w_flip_prediction['Run'+str(i+1)] = {}
        num_test_datapoints = len(test_results['Run'+str(i+1)])
        for modality in modality_list:
            # [] #store  only the predictions for that modality
            predictions_normalized_p_model = np.zeros(
                (num_test_datapoints, len(model_list)))
            for model_i in range(len(model_list)):
                # .append(np.array(predictions_train_normalized['Run'+str(i+1)][model+'_'+modality]))
                predictions_normalized_p_model[:, model_i] = predictions_test_normalized['Run'+str(
                    i+1)][model_list[model_i]+'_'+modality]
            if modality != 'FUSED':
                weights = np.transpose(np.expand_dims(
                    model_weights['Run'+str(i+1)][modality], axis=-1))
                ensemble_prediction['Run'+str(i+1)][modality] = np.multiply(
                    weights, predictions_normalized_p_model).sum(axis=1)
            else:
                ensemble_prediction['Run' +
                                    str(i+1)][modality] = predictions_normalized_p_model[:, model_i]

    for i in range(n_runs):
        for modality in modality_list:
            c_indices_test['Run'+str(i+1)]['ENS_'+modality] = concordance_index(test_results['Run'+str(
                i+1)]['True OS'], -ensemble_prediction['Run'+str(i+1)][modality], test_results['Run'+str(i+1)]['CNSR'])

    # Extend model list to include ensemble:
    model_list.append('ENS')

    # Show mean c-index per modality per model (row will be modality, column will be model)
    # Remove fused predictions with flipping. No tr error <0.5 for it to differ w.r.t. 'FUSED'
    all_array = np.zeros((n_runs, len(modality_list), len(model_list)))
    for i in range(n_runs):
        for modality_i in range(len(modality_list)):
            for model_i in range(len(model_list)):
                all_array[i, modality_i, model_i] = c_indices_test['Run' +
                                                                   str(i+1)][model_list[model_i]+'_'+modality_list[modality_i]]

    # Now row will be modality, column will be model
    means = np.mean(all_array, axis=0)

    means_df = pd.DataFrame(means, index=[modality_list], columns=[model_list])

    # Show 95% CI of c-index per modality per model (row will be modality, column will be model)
    CI = (np.std(all_array, axis=0)/np.sqrt(n_runs))*1.96

    CI_df = pd.DataFrame(CI, index=[modality_list], columns=[model_list])
    return means_df, CI_df
