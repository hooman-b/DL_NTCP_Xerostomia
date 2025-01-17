"""
IMPORTANT: if retrain=True, then make sure that the original cfg.py file (in the retrain_path folder) is
in the same directory as main.py!

Logging: https://realpython.com/python-logging/

Note:
- Set data_preproc_config.use_umcg = True

User-specific:
- features.csv
- stratified_sampling_test.csv
- config.data_dir
- config.strata_groups
- config.segmentation_structures
- config.features_dl  # Features for Deep Learning model
- data_preproc_config.endpoint
- data_preproc_config.features  # Features for Logistic Regression model
- (optional) data_preproc_config.lr_coefficients # Pretrained coefficients for Logistic Regression model
"""
import os
import sys
sys.path.append('./data_preproc')
import math
import torch
from datetime import datetime
from itertools import chain, combinations

import data_preproc.data_preproc_config as data_preproc_config
from data_preproc.data_preproc_functions import create_folder_if_not_exists

# Whether to perform quick run for checking workability of code or not
perform_test_run = False # This one is used for testing purposes.

# Set directory contexts
root_path = os.getcwd()#'//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Hooman Bahrdo/Deep_learning_datasets/Preprocessed_dataset1' #os.getcwd()
models_dir = os.path.join(root_path, 'models')
optimizers_dir = os.path.join(root_path, 'optimizers')
data_preproc_dir = os.path.join(root_path, 'data_preproc')
save_root_dir = os.path.join(root_path, 'datasets')
data_dir = os.path.join(save_root_dir, 'dataset_old_v2')
exp_root_dir = os.path.join(root_path, 'experiments', 'test')
create_folder_if_not_exists(exp_root_dir)
exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(exp_root_dir, exp_name)
exp_src_dir = os.path.join(exp_dir, 'src')
exp_models_dir = os.path.join(exp_src_dir, 'models')
exp_optimizers_dir = os.path.join(exp_src_dir, 'optimizers')
exp_data_preproc_dir = os.path.join(exp_src_dir, 'data_preproc')
exp_figures_dir = os.path.join(exp_dir, 'figures')
filename_stratified_sampling_test_csv = 'stratified_sampling_test_manual_94.csv'  # input file # Ask tomorrow
filename_stratified_sampling_full_csv = 'stratified_sampling_full_manual_94.csv'  # output file
filename_train_val_test_patient_ids_json = 'train_val_test_patient_ids.json'
filename_results_csv = 'results.csv'
filename_results_overview_csv = 'results_overview.csv'
filename_i_outputs_csv = '{i}_outputs.csv'
filename_lr_finder_png = 'lr_finder.png'
filename_model_txt = 'model.txt'
filename_best_model_pth = 'best_model.pth'
filename_results_png = 'results.png'

# Basic config
wandb_mode = 'disabled'  # 'online' | 'offline' | 'disabled' # Is a nice platform that helps visualize the model performance
torch_version = torch.__version__
seed = 4 # This seed is just for reproducibility.
nr_of_decimals = 3

# Decide which device we want to run on
gpu_condition = torch.cuda.is_available()
device = torch.device('cuda') if gpu_condition else torch.device('cpu')
cudnn_benchmark = True  # True if gpu_condition else False  # `True` will be faster, but potentially at cost of reproducibility
# This flag allows us to enable the inbuilt CuDNN auto-tuner to find the best algorithm to use for our hardware. 
# Only enable if input sizes of our network do not vary. ####### Ask about this one.

# Data config
train_frac = 0.7  # training-internal_validation-test split. The same test set will be used for Cross-Validation.
val_frac = 0.15  # training-internal_validation-test split. The same test set will be used for Cross-Validation.
sampling_type = 'stratified'  # ['random', 'stratified']. Method for dataset splitting.
perform_stratified_sampling_full = True  # (Stratified Sampling). Whether or not to recreate stratified_sampling_full.csv.
# Note: if stratified_sampling_full.csv does not exist, then we will perform stratified sampling to create the file.
strata_groups = ['xer_12']#,'xer_wk1', 'surface_bsl_dlc']#, 'Modality_adjusted', 'Loctum2']  #, 'Year_treatment_2cat']  # (Stratified Sampling). Note: order does not matter.
split_col = 'Split'  # (Stratified Sampling). Column of the stratified sampling outcome ('train', 'val', 'test').
cv_strata_groups = strata_groups  # (TODO: implement) Stratified Cross-Validation groups
cv_folds = 10 # (Cross-Validation) If cv_folds=1, then perform train-val-test-split. (For testing, I put it equal to 1)
cv_type = 'stratified'  # (Stratified CV, only if cv_folds > 1) None | 'stratified'. Stratification is performed on endpoint value.
dataset_type = 'cache'  # 'standard' | 'cache' | 'persistent'. If None, then 'standard'.
# Cache: caches data in RAM storage. Persistent: caches data in disk storage instead of RAM storage.
cache_rate = 1.0  # (dataset_type='cache')
num_workers = 8  # `4 * num_GPUs` (dataset_type='cache') # 8 or ten will be faster
pin_memory = True if num_workers > 0 else False  # Do not change
to_device = False  # if num_workers > 0 else True  # Whether or not to apply `ToDeviced()` in Dataset' transforms.
# See load_data.py. Suggestion: set True on HPC, set False on local computer.
cache_dir = os.path.join(root_path, 'persistent_cache')  # (only if dataset_type='persistent')
dataloader_type = 'standard'  # 'standard' | 'thread'. If None, then 'standard'. Thread: leverages the separate thread
# to execute preprocessing to avoid unnecessary IPC between multiple workers of DataLoader.
use_sampler = False  # Whether to use WeightedRandomSampler or not.
dataloader_drop_last = False  # (standard, thread)
segmentation_structures = data_preproc_config.structures_uncompound_list
# data_preproc_config.parotis_structures | .submand_structures | .lower_structures | .mandible_structures,
# .other_structures | .structures_uncompound_list (all)]. Structures to consider for the input segmentation map, e.g.
# only parotid and submandibulars. We can consider multiple list of structures by adding them using the '+' operator.

# Data preprocessing config: load_data.py
image_keys = ['ct', 'rtdose', 'segmentation_map']  # Do not change (subtractionct)
concat_key = 'ct_dose_seg'  # Do not change
perform_data_aug = True
rand_cropping_size = [96, 96, 96]  # (REDUNDANT)  # OLD [100, 100, 100]  # (only for training data)
input_size = [96, 96, 96]  # OLD [100, 100, 100]  # if rand_cropping_size==input_size, then no resizing will be applied
resize_mode = 'area'  # (only if perform_resize=True). Algorithm used for upsampling.
# CT
ct_a_min = -200  # OLD data_preproc_config.ct_min if data_preproc_config.perform_clipping else None
ct_a_max = 400  # OLD data_preproc_config.ct_max if data_preproc_config.perform_clipping else None
ct_b_min = 0.0
ct_b_max = 1.0
ct_clip = True
# RTDOSE
rtdose_a_min = 0
rtdose_a_max = 8000
rtdose_b_min = 0.0
rtdose_b_max = 1.0
rtdose_clip = True
# Segmentation_map  ####### Some changes here ########
# segmentation_map_a_min = 0.0
# segmentation_map_a_max = 1.0
# segmentation_map_b_min = 0.0
# segmentation_map_b_max = 1.0

# The value of element `i` in `seg_orig_labels` (e.g. value=1) will be mapped to the value of element `i`
# in `seg_target_labels` (e.g. value=2)
# Advice: use the order as ['background'] + data_preproc_config.structures_uncompound_list = \
# ['background', 'parotis_li', 'parotis_re', 'submandibularis_li', 'submandibularis_re',
#  'crico', 'thyroid', 'mandible', 'glotticarea', 'oralcavity_ext', 'supraglottic',
#  'buccalmucosa_li', 'buccalmucosa_re', 'pcm_inf', 'pcm_med', 'pcm_sup', 'esophagus_cerv']
#  Note: `+ 1` in `seg_orig_labels` because of `background`
seg_orig_labels = [x for x in range(len(data_preproc_config.structures_uncompound_list) + 1)]  # No need to change
seg_target_labels = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0.1, 0, 0.1, 0.1, 0, 0, 0, 0]  # [0] + [1] * len(data_preproc_config.structures_uncompound_list)
segmentation_map_clip = True
# D_contra_0
# d_contra_0_max = 62
# Data augmentation config (only if perform_data_aug=True)
data_aug_p = 0.5  # Probability of a data augmentation transform: data_aug_p=0 is no data aug.
data_aug_strength = 1  # Strength of data augmentation: strength=0 is no data aug (except flipping).
# Interpolation modes: 'trilinear'/'bilinear' | 'nearest'.
ct_interpol_mode_3d = 'trilinear'  # OLD 'bilinear'
rtdose_interpol_mode_3d = 'trilinear'  # OLD 'bilinear'
segmentation_interpol_mode_3d = 'nearest'  # OLD 'bilinear'
ct_dose_seg_interpol_mode_3d = 'trilinear'
ct_interpol_mode_2d = 'bilinear'
rtdose_interpol_mode_2d = 'bilinear'
segmentation_interpol_mode_2d = 'nearest'  # OLD 'bilinear'
ct_dose_seg_interpol_mode_2d = 'bilinear'
# AugMix
perform_augmix = True
mixture_width = 3  # 3 (default)
mixture_depth = [1, 3]  # [1, 3] (default)
augmix_strength = 3

# Deep Learning model config
model_name = 'resnet_dcnn_lrelu'  # ['cnn_lrelu', 'convnext_tiny', 'convnext_small', 'convnext_base',
# 'dcnn_lrelu', 'dcnn_dws_lrelu', 'dcnn_lrelu_gn', 'dcnn_lrelu_ln', 'dcnn_selu', 'efficientnet-b0', 'efficientnet-b1',
# ..., 'efficientnet-b8', 'efficientnetv2_xs', 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l',
# 'efficientnetv2_xl', 'efficientnetv2_s_selu', 'efficientnetv2_m_selu', 'efficientnetv2_l_selu',
# 'efficientnetv2_xl_selu', 'mlp', 'resnet_lrelu', 'resnet_dcnn_lrelu', 'resnet_dcnn_lrelu_v2,
# 'resnet_dcnn_dws_lrelu_v2, 'resnet_dcnn_lrelu_gn',
# 'resnet_dcnn_selu', 'resnet_mp_lrelu', 'resnet_original_relu', 'resnext_lrelu', 'resnext_original_relu'].
n_input_channels = 3  # CT, RTDOSE and Segmentation_map
features_dl = ['xer_wk1_not_at_all', 'xer_wk1_little', 'xer_wk1_moderate_to_severe', 'sex', 'age']
              #['HN35_Xerostomia_W01_not_at_all', 'HN35_Xerostomia_W01_little', 'HN35_Xerostomia_W01_moderate_to_severe',
              # 'Gender', 'Age']  # [] | data_preproc_config.features  # Should contain columns in features.csv.
resnet_shortcut = 'B'  # (resnet_original) 'A', 'B'. Pretrained resnet10_original has 'B', resnet18_original has 'A'.
filters = [8, 8, 16, 16, 32]#[6, 12, 24, 48, 96]#
kernel_sizes = [7, 5, 4, 3, 3]
strides = [[2]*3, [2]*3, [2]*3, [2]*3, [2]*3]  # strides>2 currently not supported.
# Usually len(kernel_sizes) == len(filters), but len(kernel_sizes) > len(filters) is often allowed, but then
# kernel_sizes[:len(filters)] will be used. Similarly for strides.
pad_value = 0  # (Padding) Value used for padding.
lrelu_alpha = 0.1  # (LeakyReLU) Negative slope.
pooling_conv_filters = None  # Either int or None (i.e. no pooling conv before flattening).
perform_pooling = False  # Whether to perform (Avg)Pooling or not. If pooling_conv_filters is not None, then
# (Avg)Pooling will not be applied.
linear_units = [16]
dropout_p = [0]  # Should have the same length as `linear_units`
clinical_variables_position = 0  # (Only if len(features_dl) > 0.) -1 | 0 | 1 | 2 | ... The position is in line with
# `clinical_variables_linear_units`, e.g., 0 means that the clinical variables MLP will be added to the first linear
# layer of `clinical_variables_linear_units`. Note: -1 means that the clinical variables MLP will be added to the layer
# before the first linear layer, which for most models is the flatten layer.
clinical_variables_linear_units = None  # (Only if len(features_dl) > 0.) None | list of ints
clinical_variables_dropout_p = [0]  # Should have the same length as `clinical_variables_linear_units`
use_bias = True
num_classes = 2  # Model outputs size. IMPORTANT: define `label_weights` such that len(label_weights) == num_classes.
num_ohe_classes = 2  # Size of One-Hot Encoding output.

# Optimization config
pretrained_path = None  # None, path_to_and_including_pth (e.g. 'MedicalNet/pretrain/resnet_10_23dataset.pth')
weight_init_name = 'kaiming_uniform'  # [None, 'kaiming_uniform', 'uniform', 'xavier_uniform', 'kaiming_normal',
# 'normal', 'xavier_normal', 'orthogonal']. If None, then PyTorch's default (i.e. 'kaiming_uniform', but with
# a = math.sqrt(5)). Kaiming works well if the network has (Leaky) P(ReLU) activations.
kaiming_a = math.sqrt(5)  # [math.sqrt(5) (default), lrelu_alpha]. Only used when kaiming_nonlinearity = 'leaky_relu'.
kaiming_mode = 'fan_in'  # ['fan_in' (default), 'fan_out'].
kaiming_nonlinearity = 'leaky_relu'  # ['leaky_relu' (default), 'relu', 'selu', 'linear']. When using 
# weight_init_name = kaiming_normal for initialisation with SELU activations, then nonlinearity='linear' should be used 
# instead of nonlinearity='selu' in order to get Self-Normalizing Neural Networks.
gain = 0  # [0 (default), torch.nn.init.calculate_gain('leaky_relu', lrelu_alpha)].
optimizer_name = 'ada_bound'  # ['acc_sgd', 'ada_belief', 'ada_bound', 'ada_bound_w', 'ada_hessian', 'ada_mod', 'adam',
# 'adam_w', 'apollo', 'diff_grad', 'madgrad', 'novo_grad', 'qh_adam', 'qhm', 'r_adam', 'ranger_21', 'ranger_qh',
# 'rmsprop', 'pid', 'sgd', 'sgd_w', 'swats', 'yogi']
optimizer_name_next = []  # [] | ['sgd']. Next optimizers after nr_epochs_not_improved_opt >= patience_lr.
momentum = 0  # 0 (default), 0.85, 0.9 (common), 0.95. For optimizer_name in ['rmsprop', 'sgd', 'sgd_w'].
weight_decay = 0.05  # 0 (default), 0.01 (for optimizer_name in ['adam_w', 'sgd_w']). L2 regularization penalty.
hessian_power = 1.0  # (AdaHessian)
use_lookahead = False  # (Lookahead)
lookahead_k = 5  # (Lookahead) 5 (default), 10.
lookahead_alpha = 0.5  # (Lookahead) 0.5 (default), 0.8.
loss_function_name = 'cross_entropy'  # [None, 'bce' (num_classes = 1), 'cross_entropy' (num_classes = 2), 'cross_entropy', # Ask for costume
# 'dice', 'f1', 'ranking', 'soft_auc', 'custom']. Note: if 'bce', then also change label_weights to list of 1 element. # use costum again
# Note: model output should be logits, i.e. NO sigmoid() (BCE) nor softmax() (CE) applied.
loss_weights = [1, 0, 1, 1, 0, 0]  # [1/6, 1/6, 1/6, 1/6, 1/6, 1/6].
# (loss_function_name='custom') list of weight for [ce, dice, f1, l1, ranking, soft_auc].
label_weights = [1, 1.5]  # [1, 1] (ce) | [1] (bce) | w_jj = (1 - /beta) / (1 - /beta^{n_samples_j}) (\beta = 0.9, 0.99) |
# wj=n_samples / (n_classes * n_samplesj). Rescaling (relative) weight given to each class, has to be list of size C.
loss_reduction = 'mean'
label_smoothing = 0  # (LabelSmoothing) If 0, then no label smoothing will be applied. Currently only supported for
# loss_function_name='cross_entropy'.
scheduler_name = 'cosine'  # [None, 'cosine', 'cyclic', 'exponential', 'step', 'lr_finder', 'reduce_lr', 'batch_size',
# 'manual_lr']. If None, then no scheduler will be applied.
grad_max_norm = None  # (GradientClipping) Maximum norm of gradients. If None, then no grad clipping will be applied.
lr = 1e-4  # 0.0004750810162102798  # Redundant if perform_lr_finder=True.
lr_finder_lower_lr = 1e-6  # (LearningRateFinder) Maximum learning rate value to try.
lr_finder_upper_lr = 1e-3  # (LearningRateFinder) Minimum learning rate value to try.
lr_finder_num_iter = 0  # (LearningRateFinder) Number of learning rates to try within [lower_lr, upper_lr].
warmup_batches = 0  # (LearningRateWarmUP) Number of warmup batches. If warmup = 0, then no warmup. Does not work with
# manual schedulers (e.g. 'manual_lr', see misc.get_scheduler()).
T_0 = 8  # (CosineAnnealingWarmRestarts) Number of epochs until a restart.
T_mult = 1  # (CosineAnnealingWarmRestarts) A factor increases after a restart. Default: 1.
eta_min = 1e-8  # (CosineAnnealingWarmRestarts) Minimum learning rate. Default: 0.
base_lr = 1e-7  # (CylicLR, only if perform_lr_finder = False, see get_scheduler()) Minimum and starting learning rate.
max_lr = 1e-4  # (CylicLR, only if perform_lr_finder = False, see get_scheduler()) Maximum learning rate.
step_size_up = 79 * 15  # (CyclicLR) Number of training iterations in the increasing half of a cycle.
gamma = 0.95  # (ExponentialLR, StepLR) Multiplicative factor of learning rate decay every epoch (ExponentialLR) or
# step_size (StepLR).
step_size = 15  # (StepLR) Decays the learning rate by gamma every step_size epochs.
factor_lr = 0.5  # (Reduce_LR) Factor by which the learning rate will be updated: new_lr = lr * factor.
min_lr = 1e-8  # (Reduce_LR) Minimum learning rate allowed.
patience_lr = 7  # (LR_Finder, Reduce_LR) Perform LR Finder after this number of consecutive epochs without
# internal validation improvements.
patience_opt = 9  # Use next optimizer in 'optimizer_name_next' after nr_epochs_not_improved_opt >= patience_opt.
manual_lr = [1e-3, 1e-5, 1e-6]  # (Manual_LR) LR per epoch, if epoch > len(manual_lr), then use lr = manual_lr[-1].
# Note: if perform_lr_finder = True, then the first value of manual_lr will not be used!

# Training config
nr_runs = 1
max_epochs = 20
batch_size = 8
max_batch_size = 16
eval_interval = 1
patience = 10  # (EarlyStopping): stop training after this number of consecutive epochs without
# internal validation improvements.

# Plotting config
plot_interval = 10
max_nr_images_per_interval = 1
figsize = (12, 12)

if perform_test_run:
    lr_finder_num_iter = 0
    n_samples = 40
    nr_runs = 1
    max_epochs = 1
    train_frac = 0.6 # What???!!!Why???!!!
    val_frac = 0.2
    cv_folds = 2
    batch_size = 3
    num_workers = 0
    pin_memory = False
    plot_interval = 1
    max_nr_images_per_interval = 5

##### Optuna (Hyperparameter Tunning)#####
# Add optuna initializing variables
optuna_path_pickles = os.path.join(exp_root_dir, 'pickles')
optuna_path_figures = os.path.join(exp_root_dir, 'figures')

# Optuna Basic parameters
optuna_batch_size_list = [2, 4, 8, 16]
optuna_filters_list = [[16, 16, 32, 32, 64], [8, 8, 16, 32, 64], [16, 16, 32, 64, 128], [16, 16, 16, 32, 32], [8, 8, 16, 16, 32], [8, 8, 8, 16, 16]]
optuna_min_kernel_size = 3
optuna_max_kernel_size = 7
optuna_linear_units_size = [1, 2, 3]
optuna_clinical_variables_linear_units_size = [1, 2, 3]
optuna_linear_units_list = [8, 16, 32, 64]
optuna_clinical_variables_linear_units_list = [8, 16, 32, 64]
T0 = [10,20,30,40]


def make_clinical_features(features):
    # Make all the possible combination of the elements
    result_list = [list(combination) for r in range(1, len(features) + 1)
                for combination in combinations(features, r)]

    # Adding the empty sublist
    result_list.insert(0, [])
    return result_list


optuna_features_dl_list = [[]]#[['xer_wk1_not_at_all', 'xer_wk1_little', 'xer_wk1_moderate_to_severe', 'sex', 'age']] #[[] , ['xer_wk1_not_at_all', 'xer_wk1_little', 'xer_wk1_moderate_to_severe', 'sex', 'age']]
print(optuna_features_dl_list)


# Write about the changes I should make in the code to use the WeeklyCT Images.
# Line 537 load_data:
# change 'MapLabelValued' to 'ScaleIntensityRanged' and instead of 'segmentation maps',
# I need to use 'weeklyCTs'
# 
# Line 173 load_data:
# change 'segmentation_map' into 'weeklyct' label and alse the direction to weeklyCTs.
# it may change other properties. I will add them here as well.
#
# Line 50 process_data:
# I add 'ScaleIntensityRange' for weeklyCTs here and add it to the rest of the code,
# But for segmentation_map nothing should be added.
#
# Line 226 process_data
# I changed the augmentation method from segmentation map to weeklyCTs and I am using
# bilinear as used for CT.
#
# Line 440 main
# Change some ioutputs in the result part. instead of using segmentation maps,
# I am using information for weeklyCTs.

# Changes for subtraction model
# Line 537 load_data:
# Here I change 'ct.npy' to 'subtractionct.npy'
#
# Line 173 load_data:
# change 'ct' into 'subtractionct' label and also change the direction.
#
# Line 104 in config
# change the key image



# WEEKLYCT PARAMETERS
# WEEKLYCT
weeklyct_a_min = -200  
weeklyct_a_max = 400  
weeklyct_b_min = 0.0
weeklyct_b_max = 1.0
weeklyct_clip = True

weeklyct_interpol_mode_2d = 'bilinear'
weeklyct_interpol_mode_3d = 'trilinear'