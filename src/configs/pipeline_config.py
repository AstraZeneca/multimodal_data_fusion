import collections, copy

def pipeline_config(exptname: str = None):
  """
  Method to define the configuration parameters

  Parameters
  ----------
    exptname : STRING, optional

  Returns
  -------
  expt_config : DICT
    Configuration dictionary
  expt_name : STRING
    experiment name  
  """
  if exptname==None:
    exptname = 'default'

  experiments = collections.defaultdict(dict)
    
  #---------------------------------------------------------------------------
  # Setting up the child dictionaries
  #---------------------------------------------------------------------------
  
  # Modalities
  modalities_dict = {}
  modalities_dict['CLINICAL'] = '_clin'
  modalities_dict['PYRADIOMICS'] = '_pr'

  # Preprocessing settings
  preprocess_settings = {}
  preprocess_settings['modality'] = ['PYRADIOMICS']  # List of modalities to preprocess
  preprocess_settings['type'] = ['minmax_scale']  # 'normalization' is any of 'standardize', 'robust_scale' or 'minmax_scale' for continuous features
  
  # settings
  settings = {}
  settings['modalities'] = copy.deepcopy(modalities_dict)
  settings['stratify'] = False
  settings['strat_columns'] = ['STUDY', 'PDL1_STATUS']
  settings['split_mode'] = 'subject'  # Split per-row ('row') or per subject ID ('subject') or predefined ('predefined') -- A fraction of frac_train of the rows / subjects will be used for training, the rest for testing
  settings['train_fraction'] = 0.8 #Fraction of data to be used as the training set for feature selection & modelling:
  settings['cohorts'] = '' # Path to predined splits if split_mode == predefined
  settings['preprocessing'] = copy.deepcopy(preprocess_settings)

  # Consistency settings
  fs_consistency_settings = {}
  fs_consistency_settings['n_runs'] = 5  # Try 1 & 1 or 0 & 1
  fs_consistency_settings['consistency_threshold'] = 2
  
  # Set the unsupervised feature selection settings
  unsupervised_fs_settings = {}
  unsupervised_fs_settings['drop_qconstant'] = 0  # 1 to remove low variance features, 0 to not do so
  unsupervised_fs_settings['threshold_qconst'] = 0.05  # # Variance threshold below which to consider feature quasi-constant (low-variance)
  unsupervised_fs_settings['handle_correlated'] = 'ignore'  # 'drop' to remove correlated features, 'group' to average them
  unsupervised_fs_settings['corr_threshold_high'] = 0.8  #threshold on absolute Pearson correlation above which to consider 'highly correlated'

  # Set the unsupervised feature selection settings
  supervised_fs_settings = {}
  supervised_fs_settings['discretization_bins'] = 10  #Some feature selection methods require discretized features &/or targets; this specifies how many bins to use
  supervised_fs_settings['target_fs'] = 'discrete'  # Options: 'discrete', 'continuous', 'median_binarized'
  supervised_fs_settings['method'] = 'MIM'  # Method to use; Options: MIM, ICAP, DISR, MRMR, JMI, CMIM, CIFE, MIFS --Suggested are JMI or CMIM
  supervised_fs_settings['K_top'] = 50  # Top K most informative features to select in each featureset (if bootstraps are used to select most consistent among these, this number will be further reduced)

  # Set the feature selection settings
  feature_selection_settings = {}
  feature_selection_settings['modality_handling'] = 'per_modality' #Options: 'per_modality' / 'joint'
  feature_selection_settings['subset_selection_mode'] = 'feature_selection' # Options: 'feature_selection' / 'none'
  feature_selection_settings['consistency_options'] = copy.deepcopy(fs_consistency_settings)
  feature_selection_settings['unsupervised'] = copy.deepcopy(unsupervised_fs_settings)
  feature_selection_settings['supervised'] = copy.deepcopy(supervised_fs_settings)
  
  # Set the model hyperparameters
  model_hyperparameters = {}
  model_hyperparameters['FRAC_TRAIN_AE'] = 0.8
  model_hyperparameters['HIDDEN_DIM'] = 64
  model_hyperparameters['LATENT_SPACE'] = 30
  model_hyperparameters['EPOCHS'] = 100
  model_hyperparameters['ES_PATIENCE'] = 20
  model_hyperparameters['BATCH_SIZE'] = 2

  # Set the dimensionality reduction settings
  dimensionality_reduction_settings = {}
  dimensionality_reduction_settings['model'] = None  #'autoencoder' or 'none'
  dimensionality_reduction_settings['modality_handling_mode_dr'] = 'joint'  #'joint' or 'per_modality'
  dimensionality_reduction_settings['excluded_modalities'] = []  # list of modalities to exclude from transformation; can leave empty
  dimensionality_reduction_settings['model_hyperparameters'] = copy.deepcopy(model_hyperparameters)

  # Set the user options
  user_options_settings = {}
  user_options_settings['verbose'] = 1  #1 to detail outcome of each step, 0 only prints some details (e.g. general step, dimensions per step, final model c-indices, etc.)
  user_options_settings['save_selected'] = 0 #1 to save selected featuresets (also train/test set), 0 to not do so -- TODO: Refine this to also save models, performances etc.
  user_options_settings['return_feature_importance_list'] = 1 #1 to produce the feature importance list for any model trained, 0 not to do so
  user_options_settings['shuffle_targets'] = 0 # ONLY FOR SANITY CHECKS! 1 will randomly shuffle the targets, breaking predictiveness of features -for now only used to check feature selection

  # Data settings
  data_settings = {}
  data_settings['missingdata_mode'] = ["drop", None]  # Options are drop all, impute with zeros, mean or median
  data_settings['pdl1'] = 'PDL1_STATUS_clin'
  data_settings['subject'] = 'SUBJID'
  data_settings['targets'] = ['OS', 'OS.time']
  data_settings['features'] = [["COMBO_clin", "SEX_F_clin", "ECOG_clin", "SMOKE_NEVER_clin", "BLBMI_clin", "ALB_clin", "NEUT_clin", "LDH_clin",
                              "AST_clin", "GGT_clin", "EOS_clin", "LYM_clin", "ALP_clin", "BASO_clin", "CA_clin", "CL_clin", "HCT_clin", "K_clin",
                              "MG_clin", "MONO_clin", "PLAT_clin", "TSH_clin", "GLUC_clin", "PROT_clin", "BLTUMSZ_clin"], []]

  # Cohorts settings
  cohortsFilter_settings = {}
  cohortsFilter_settings['CohortsAvailable']     = False
  cohortsFilter_settings['UseCohorts']           = False
  cohortsFilter_settings['ExcludeCohorts']       = False
  cohortsFilter_settings['CohortTable']          = None  # Enter the path to the cohort csv. Eg: '/projects/other/pmb_radx/temp/cohorts.csv'
  cohortsFilter_settings['CohortColumn']         = 'COHORT'  # Column name with cohort information
  cohortsFilter_settings['CohortSplits']         = ['training', 'validation', 'test']  # Column name with cohort information
  cohortsFilter_settings['CohortAttributes']     = ['SUBJID']  # Column names to find unique combinations

  # Modelling settings
  modelling_settings = {}
  modelling_settings['num_runs'] = 5  # Number of full runs of entire pipeline on a different train/test split
  modelling_settings['subsets'] = [['PYRADIOMICS']] # List of lists - each inner list contains modalities (strings) to be included in a single experiment
  modelling_settings['models'] = ['RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN'] #'RSF', 'CPH-GB', 'CLS-GB', 'CPH-L2', 'CPH-EN', 'DEEP-SURV': List containing 1 or more of the models below; leave empty to skip
  modelling_settings['pdl1'] = False  # set True to include the same
  modelling_settings['task'] = "regression"  # OPtions are regression/classification
  modelling_settings['threshold'] = 365  # Threshold to convert regression to classification
  modelling_settings['sensitivity_analysis'] = True  # Sensitivity analysis or cohorts
  modelling_settings['data_characteristics'] = copy.deepcopy(data_settings)
  modelling_settings['cohorts_settings'] = copy.deepcopy(cohortsFilter_settings)

  # Archival Settings
  ExptOutputArchivalSettings = {}
  ExptOutputArchivalSettings['Enabled'] = True # Output archival to disc is enabled by default
  ExptOutputArchivalSettings['WhenDisabledLoadFromExpt'] = 'default' #name of experiment whose output datadict you want to load (instead of re-creating)

  #---------------------------------------------------------------------------
  #DEFAULT -- to be used as template for any custom experiments (see examples below)
  #---------------------------------------------------------------------------
  experiments['default']                                      = {}
  experiments['default']['ExptDescription']                   = 'Default experiment with all radiomic features enabled and no clinical variables'
  experiments['default']['user_options_settings']             = copy.deepcopy(user_options_settings)
  experiments['default']['settings']                          = copy.deepcopy(settings)
  experiments['default']['feature_selection_settings']        = copy.deepcopy(feature_selection_settings)
  experiments['default']['dimensionality_reduction_settings'] = copy.deepcopy(dimensionality_reduction_settings)
  experiments['default']['modelling_settings']                = copy.deepcopy(modelling_settings)
  experiments['default']['ArchivalSettings']                  = copy.deepcopy(ExptOutputArchivalSettings)

  #---------------------------------------------------------------------------
  #EXPERIMENT 000001 -- Survival modelling  with clinical variables
  #---------------------------------------------------------------------------
  experiments['000001']                                   = {}
  experiments['000001']                                   = copy.deepcopy(experiments['default'])
  experiments['000001']['ExptDescription']                = 'experiment with all radiomic featues and clinical variables enabled'
  experiments['000001']['modelling_settings']['subsets']  = [['CLINICAL', 'PYRADIOMICS']]   

  #---------------------------------------------------------------------------
  #EXPERIMENT 000002 -- Survival modelling with stratification
  #---------------------------------------------------------------------------
  experiments['000002']                                                 = {}
  experiments['000002']                                                 = copy.deepcopy(experiments['default'])
  experiments['000002']['ExptDescription']                              = 'experiment with all radiomic featues for regression task and stratified split'
  experiments['000002']['settings']['stratify']                         = True

  #---------------------------------------------------------------------------
  #EXPERIMENT 000003 -- Survival modelling with pre-defined cohorts
  #---------------------------------------------------------------------------
  experiments['000003']                                                               = {}
  experiments['000003']                                                               = copy.deepcopy(experiments['default'])
  experiments['000003']['ExptDescription']                                            = 'experiment with all radiomic featues for regression task'
  experiments['000003']['modelling_settings']['sensitivity_analysis']                 = False
  experiments['000003']['modelling_settings']['cohorts_settings']['CohortsAvailable'] = True
  experiments['000003']['modelling_settings']['cohorts_settings']['UseCohorts']       = True
  experiments['000003']['modelling_settings']['cohorts_settings']['CohortTable']      = '/projects/other/pmb_radx/durva_nsclc/build/modelling_sets/cohorts.csv'

  

  if not exptname in experiments:
      exptname = 'default'

  expt_config = copy.deepcopy(experiments[exptname])

  return expt_config, exptname
