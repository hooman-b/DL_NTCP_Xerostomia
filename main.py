"""
Script for developing deep learning models via cross-validation, ensembling the models (mean prediction), and
independently testing the final ensemble model.

TODO: Add loop to evaluate different loss function, optimizers and models.
"""
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

import os
import time
import torch
import optuna
import plotly
import joblib
import shutil
import random
import numpy as np
import matplotlib

# import wandb

matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from monai.utils import set_determinism
from monai.metrics import MSEMetric, ROCAUCMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
)

# Load local functions
import config
import misc
import load_data
import process_data
import data_preproc.data_preproc_config as data_preproc_config

from data_preproc.data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from data_preproc.checks.check_data_preproc import plot_multiple_arrs
from models.logistic_regression import run_logistic_regression

# Handle increase the file descriptor error
import torch.multiprocessing as mp

# Change the multiprocessing strategy to 'file_system'
mp.set_sharing_strategy('file_system')


def initialize(torch_version, device, create_folders=True):
    """
    Set up experiment and save the experiment path to be able to save results.

    Args:
        torch_version:
        device:
        create_folders:

    Returns:

    """
    if create_folders:
        # Create experiment folder and subfolders if they do not exist yet
        exp_dir = config.exp_dir
        exp_src_dir = config.exp_src_dir
        exp_models_dir = config.exp_models_dir
        exp_optimizers_dir = config.exp_optimizers_dir
        exp_data_preproc_dir = config.exp_data_preproc_dir
        exp_figures_dir = config.exp_figures_dir
        optuna_path_pickles = config.optuna_path_pickles
        optuna_path_figures = config.optuna_path_figures

        for p in [exp_dir, exp_src_dir, exp_models_dir, exp_optimizers_dir, exp_data_preproc_dir, exp_figures_dir, optuna_path_pickles, optuna_path_figures]:
            create_folder_if_not_exists(p)

        # Fetch folders and files that should be copied to exp folder
        root_path = config.root_path
        models_dir = config.models_dir
        optimizers_dir = config.optimizers_dir
        data_preproc_dir = config.data_preproc_dir
        src_files = [x for x in os.listdir(root_path) if
                     (x.endswith('.py') or x == data_preproc_config.filename_exclude_patients_csv)]

        # Copy src files to exp folder
        for f in src_files:
            copy_file(src=os.path.join(root_path, f), dst=os.path.join(exp_src_dir, f))
        # Copy folders to exp folder
        copy_folder(src=models_dir, dst=exp_models_dir)
        copy_folder(src=optimizers_dir, dst=exp_optimizers_dir)
        copy_folder(src=data_preproc_dir, dst=exp_data_preproc_dir)

        # Logger output filename
        output_filename = os.path.join(exp_dir, 'log.txt')
    else:
        output_filename = None

    # Initialize logger
    logger = Logger(output_filename=output_filename)

    # Print variables
    logger.my_print('Device: {}.'.format(device))
    logger.my_print('Torch version: {}.'.format(torch_version))
    logger.my_print('Torch.backends.cudnn.benchmark: {}.'.format(torch.backends.cudnn.benchmark))

    return logger


def validate(model, dataloader, mean, std, mode,
             loss_function, loss_function_name, sigmoid_act, softmax_act, auc_metric, mse_metric,
             to_onehot, num_classes, model_name, exp_dir,
             device,
             logger, save_outputs=True):
    """
    Validate the model.

    Args:
        model:
        dataloader:
        mean: normalization: list of means, one value for each channel (first value for CT, second value for RTDOSE)
        std: normalization: list of stds, one value for each channel (first value for CT, second value for RTDOSE)
        mode (str): for printing, either 'internal validation' or 'test'
        logger:

    Returns:

    """
    # Initialize variable
    model.eval()
    mode = mode.capitalize()
    loss_value = 0
    num_iterations = len(dataloader)
    patient_ids = list()

    with torch.no_grad():
        y_pred = torch.as_tensor([], dtype=torch.float32, device=device)
        y = torch.as_tensor([], dtype=torch.int8, device=device)
        for data in dataloader:
            # Load data
            inputs, features, labels = (
                data['ct_dose_seg'].to(device),
                data['features'].to(device),
                data['label'].to(device),
            )

            # Preprocess inputs and features
            # NOTE: these operations should also be performed in lr_finder.py
            inputs = process_data.preprocess_inputs(inputs=inputs,
                                                    ct_mean=mean[0], ct_std=std[0],
                                                    rtdose_mean=mean[1], rtdose_std=std[1])
            features = process_data.preprocess_features(features=features)

            print(features)
            # Make predictions
            outputs = model(x=inputs, features=features.float())

            try:
                # Cross-Entropy, Ranking, Custom
                loss = loss_function(outputs, labels)
            except:
                # BCE
                loss = loss_function(outputs, torch.reshape(labels, outputs.shape).to(outputs.dtype))

            # Evaluate model (internal validation set)
            loss_value += loss.item()
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, labels], dim=0)

            # Save patient_ids
            patient_ids += data['patient_id']

        # Averaging internal validation loss
        loss_value /= num_iterations
        # loss_value = round(loss_value, nr_of_decimals)

        # Compute internal validation AUC and MSE
        if loss_function_name in ['bce']:
            y_pred_list = misc.proba(sigmoid_act(y_pred))
        else:
            y_pred_list = [softmax_act(i) for i in y_pred]
        y_list = [to_onehot(i) for i in y]
        auc_value = misc.compute_auc(y_pred_list=y_pred_list, y_true_list=y_list, auc_metric=auc_metric)
        mse_value = misc.compute_mse(y_pred_list=y_pred_list, y_true_list=y_list, mse_metric=mse_metric)

        if mode.lower() == 'test':
            # Save outputs to csv
            if save_outputs:
                mode_list = ['test'] * len(patient_ids)
                misc.save_predictions(patient_ids=patient_ids, y_pred_list=y_pred_list, y_true_list=y_list,
                                      mode_list=mode_list, num_classes=num_classes, model_name=model_name,
                                      exp_dir=exp_dir, logger=logger)
            return loss_value, mse_value, auc_value, patient_ids, y_pred_list, y_list

        # Prints
        logger.my_print(f'{mode} loss: {loss_value:.3f}.')
        logger.my_print(f'{mode} AUC: {auc_value:.3f}.')
        logger.my_print(f'{mode} MSE: {mse_value:.3f}.')

        return loss_value, mse_value, auc_value


def train(model, train_dataloader, val_dataloader, test_dataloader, mean, std, optimizer, optimizer_name_next,
          dl_class, train_dl_args_dict,
		  num_classes,
          max_epochs, perform_augmix, mixture_width, mixture_depth, augmix_strength, optimizer_name,
          grad_max_norm, scheduler_name, scheduler, loss_function_name, sigmoid_act, softmax_act, to_onehot,
          patience, patience_lr, patience_opt, loss_function, auc_metric, mse_metric, data_aug_p, data_aug_strength,
          max_batch_size, manual_lr,
          momentum, weight_decay, hessian_power, num_batches_per_epoch, use_lookahead, lookahead_k, lookahead_alpha,
          device,
		  model_name, exp_dir, 
          logger):
    """
    Train the model on the combined dataset, generate images, and save the images and the models.

    Args:
        model: PyTorch model
        train_dataloader: training dataloader
        val_dataloader: internal validation dataloader
        test_dataloader:
        mean: normalization: list of means, one value for each channel (first value for CT, second value for RTDOSE)
        std: normalization: list of stds, one value for each channel (first value for CT, second value for RTDOSE)
        optimizer:
        optimizer_name_next:
        dl_class:
        train_dl_args_dict:
        logger: logger

    Returns:
        None
    """
    # Initiate variables
    best_val_loss_value = np.inf
    best_val_mse_value = np.inf
    best_val_auc_value = -1
    train_loss_values_list = list() if max_epochs > 0 else [None]
    train_mse_values_list = list() if max_epochs > 0 else [None]
    train_auc_values_list = list() if max_epochs > 0 else [None]
    val_loss_values_list = list() if max_epochs > 0 else [None]
    val_mse_values_list = list() if max_epochs > 0 else [None]
    val_auc_values_list = list() if max_epochs > 0 else [None]
    test_loss_values_list = list() if max_epochs > 0 else [None]
    test_mse_values_list = list() if max_epochs > 0 else [None]
    test_auc_values_list = list() if max_epochs > 0 else [None]
    lr_list = list() if max_epochs > 0 else [None]
    batch_size_list = list() if max_epochs > 0 else [None]
    nr_epochs_not_improved = 0  # (EarlyStopping)
    nr_epochs_not_improved_lr = 0  # (LR_Finder, Reduce_LR)
    nr_epochs_not_improved_opt = 0  # (Next optimizer)

    train_num_iterations = len(train_dataloader)
    logger.my_print('Number of training iterations per epoch: {}.'.format(train_num_iterations))
    epoch = 0
    best_epoch = 0
    for epoch in range(max_epochs):
        # Print training stats
        logger.my_print(f'Epoch {epoch + 1}/{max_epochs}...')
        for param_group in optimizer.param_groups:
            logger.my_print('Learning rate: {}.'.format(param_group['lr']))
            lr_list.append(param_group['lr'])
        cur_batch_size = train_dataloader.batch_size
        logger.my_print('Batch size: {}.'.format(cur_batch_size))
        batch_size_list.append(cur_batch_size)

        # Initiate training
        model.train()
        train_loss_value = 0
        train_y_pred = torch.as_tensor([], dtype=torch.float32, device=device)
        train_y = torch.as_tensor([], dtype=torch.int8, device=device)

        for i, batch_data in tqdm(enumerate(train_dataloader)):
            # Load data
            train_inputs, train_features, train_labels = (
                batch_data['ct_dose_seg'].to(device),
                batch_data['features'].to(device),
                batch_data['label'].to(device)
            )
            print(batch_data['patient_id'])
            if perform_augmix:
                for b in range(len(train_inputs)):
                    # Generate a random 32-bit integer
                    seed_b = random.getrandbits(32)
                    train_inputs[b] = process_data.aug_mix(arr=train_inputs[b], mixture_width=mixture_width,
                                                           mixture_depth=mixture_depth, augmix_strength=augmix_strength,
                                                           device=device, seed=seed_b)

            # Preprocess inputs and features
            # NOTE: these operations should also be performed in lr_finder.py ##### Maybe some changes here, for my model. #####
            train_inputs = process_data.preprocess_inputs(inputs=train_inputs, ct_mean=mean[0], ct_std=std[0],
                                                          rtdose_mean=mean[1], rtdose_std=std[1])
            train_features = process_data.preprocess_features(features=train_features)

            # print('hiiiiiiiiiiiiiiiiiii', train_features)
            # train_inputs = train_inputs.to(torch.long)
            # train_features = train_features.to(torch.long)

            # Zero the parameter gradients and make predictions
            optimizer.zero_grad(set_to_none=True)
            train_outputs = model(x=train_inputs, features=train_features.float())

            # Calculate loss
            try:
                # Cross-Entropy, Ranking, Custom
                train_loss = loss_function(train_outputs, train_labels)
            except:
                # BCE
                train_loss = loss_function(train_outputs,
                                           torch.reshape(train_labels, train_outputs.shape).to(train_outputs.dtype))

            if optimizer_name in ['ada_hessian']:
                # https://github.com/pytorch/pytorch/issues/4661
                # https://discuss.pytorch.org/t/how-to-backward-the-derivative/17662?u=bpetruzzo
                # Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient
                # which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this.
                # However, if we do use backward(create_graph=True), then we have to make sure to reset the
                # .grad fields of our parameters to None after use to break the cycle and avoid the leak.
                train_loss.backward(create_graph=True)
                # torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
            else:
                train_loss.backward()

            # Perform gradient clipping
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            if grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_max_norm)

            # Update model weights
            optimizer.step()

            # Reset the .grad fields of our parameters to None after use to break the cycle and avoid the memory leak
            if optimizer_name in ['ada_hessian']:
                for p in model.parameters():
                    p.grad = None

            # Scheduler: step() called after every batch update
            # Note: some schedulers generally update the LR every epoch instead of every batch, but epoch-wise update
            # will mess up LearningRateWarmUp. Batch-wise LR update is always valid and works with LearningRateWarmUp.
            if scheduler_name in ['cosine', 'exponential', 'step']:
                scheduler.step(epoch + (i + 1) / train_num_iterations)
            elif scheduler_name in ['cyclic']:
                scheduler.step()

            # Evaluate model (training set)
            train_loss_value += train_loss.item()
            train_y_pred = torch.cat([train_y_pred, train_outputs], dim=0)
            train_y = torch.cat([train_y, train_labels], dim=0)

        # Averaging training loss
        train_loss_value /= train_num_iterations

        # Compute training AUC and MSE
        if loss_function_name in ['bce']:
            train_y_pred_list = misc.proba(sigmoid_act(train_y_pred))
        else:
            train_y_pred_list = [softmax_act(i) for i in train_y_pred]
        train_y_list = [to_onehot(i) for i in train_y]
        train_auc_value = misc.compute_auc(y_pred_list=train_y_pred_list, y_true_list=train_y_list,
                                           auc_metric=auc_metric)
        train_mse_value = misc.compute_mse(y_pred_list=train_y_pred_list, y_true_list=train_y_list,
                                           mse_metric=mse_metric)

        # Prints
        logger.my_print(f'Training loss: {train_loss_value:.3f}.')
        logger.my_print(f'Training AUC: {train_auc_value:.3f}.')
        logger.my_print(f'Training MSE: {train_mse_value:.3f}.')

        train_loss_values_list.append(train_loss_value)
        train_mse_values_list.append(train_mse_value)
        train_auc_values_list.append(train_auc_value)

        if (epoch + 1) % config.eval_interval == 0:
            # Perform internal validation  
            val_loss_value, val_mse_value, val_auc_value = validate(model=model, dataloader=val_dataloader,
                                                                    mean=mean, std=std,
                                                                    mode='internal validation', 
                                                                    loss_function=loss_function, loss_function_name=loss_function_name,
                                                                    sigmoid_act=sigmoid_act, softmax_act=softmax_act, auc_metric=auc_metric, mse_metric=mse_metric,
                                                                    to_onehot=to_onehot, num_classes=num_classes, model_name=model_name, exp_dir=exp_dir,
                                                                    device=device,
                                                                    logger=logger,
                                                                    save_outputs=True)
            val_loss_values_list.append(val_loss_value)
            val_loss_values_list.append(val_loss_value)
            val_mse_values_list.append(val_mse_value)
            val_auc_values_list.append(val_auc_value)

            # # Perform independent testing (Redundant)
            # test_loss_value, test_mse_value, test_auc_value = validate(model=model, dataloader=test_dataloader,
            #                                                            mean=mean, std=std,
            #                                                            mode='independent testing', logger=logger,
            #                                                            save_outputs=True)
            # test_loss_values_list.append(test_loss_value)
            # test_mse_values_list.append(test_mse_value)
            # test_auc_values_list.append(test_auc_value)
            test_loss_values_list = []
            test_mse_values_list = []
            test_auc_values_list = []

            # Select model
            if val_auc_value > best_val_auc_value:
                best_val_loss_value = val_loss_value
                best_val_auc_value = val_auc_value
                best_val_mse_value = val_mse_value
                best_epoch = epoch + 1
                nr_epochs_not_improved = 0
                nr_epochs_not_improved_lr = 0
                nr_epochs_not_improved_opt = 0
                torch.save(model.state_dict(), os.path.join(config.exp_dir, config.filename_best_model_pth))
                logger.my_print('Saved new best metric model.')
            else:
                nr_epochs_not_improved += 1
                nr_epochs_not_improved_lr += 1
                nr_epochs_not_improved_opt += 1

            logger.my_print(f'Best internal validation AUC: {best_val_auc_value:.3f} at epoch: {best_epoch}.')
            logger.my_print(
                f'Corresponding internal validation loss: {best_val_loss_value:.3f} at epoch: {best_epoch}.')
            logger.my_print(f'Corresponding internal validation MSE: {best_val_mse_value:.3f} at epoch: {best_epoch}.')

            # W&B logging
            # wandb.log({'epoch': epoch + 1,
            #            'train_loss_value': train_loss_value,
            #            'train_mse_value:': train_mse_value,
            #            'train_auc_value': train_auc_value,
            #            'val_loss_value': val_loss_value,
            #            'val_mse_value': val_mse_value,
            #            'val_auc_value': val_auc_value,
            #            'batch_size': batch_size_list[-1],
            #            'learning_rate': lr_list[-1]})

        if (epoch + 1 == 1) or ((epoch + 1) % config.plot_interval == 0):
            # Plot for random patients
            plot_inputs = train_inputs
            for patient_idx in range(min(config.max_nr_images_per_interval, len(plot_inputs))):
                arr_list = [plot_inputs[patient_idx][0].cpu() * std[0] + mean[0],
                            plot_inputs[patient_idx][1].cpu() * std[1] + mean[1],
                            plot_inputs[patient_idx][2].cpu()]
                
                # ########### some changes in the output ###########
                cmap_list = [data_preproc_config.ct_cmap, data_preproc_config.rtdose_cmap,
                             data_preproc_config.segmentation_cmap]
                # cmap_list = [data_preproc_config.ct_cmap, data_preproc_config.rtdose_cmap,
                #               data_preproc_config.weeklycd_cmap]
            
                # vmin_list = [0, 0, 0]
                vmin_list = [config.ct_a_min, config.rtdose_a_min, 0]
                # vmin_list = [config.ct_a_min, config.rtdose_a_min, config.weeklyct_a_min]

                # vmax_list = [1, config.data_preproc_config.rtdose_vmax/config.rtdose_a_max, 1]
                vmax_list = [config.ct_a_max, config.data_preproc_config.rtdose_vmax, 1]
                # vmax_list = [config.ct_a_max, config.data_preproc_config.rtdose_vmax, config.weeklyct_a_max]

                ticks_steps_list = [vmax_list[0] - vmin_list[0], vmax_list[1] - vmin_list[1], 1]
                # ticks_steps_list = [vmax_list[0] - vmin_list[0], vmax_list[1] - vmin_list[1], vmax_list[2] - vmin_list[2]]

                segmentation_map_list = [arr_list[2] >= 0.5, None, None]
                # segmentation_map_list = None

                colorbar_title_list = [data_preproc_config.ct_colorbar_title, data_preproc_config.rtdose_colorbar_title,
                                       data_preproc_config.segmentation_colorbar_title]
                # colorbar_title_list = [data_preproc_config.ct_colorbar_title, data_preproc_config.rtdose_colorbar_title,
                #                        data_preproc_config.weeklyct_colorbar_title]

                # Overlay
                overlay_list = [None, None, None]
                alpha_overlay_list = [None, None, None]
                cmap_overlay_list = [None, None, None]
                vmin_overlay_list = [None, None, None]
                vmax_overlay_list = [None, None, None]
                plot_multiple_arrs(arr_list=arr_list, overlay_list=overlay_list, alpha_overlay_list=alpha_overlay_list,
                                   nr_images=data_preproc_config.plot_nr_images_multiple,
                                   figsize=data_preproc_config.figsize_multiple, cmap_list=cmap_list,
                                   cmap_overlay_list=cmap_overlay_list, colorbar_title_list=colorbar_title_list,
                                   filename=os.path.join(config.exp_figures_dir,
                                                         'epoch_{}_idx_{}.png'.format(epoch + 1, patient_idx)),
                                   vmin_list=vmin_list, vmin_overlay_list=vmin_overlay_list, vmax_list=vmax_list,
                                   vmax_overlay_list=vmax_overlay_list, ticks_steps_list=ticks_steps_list,
                                   segmentation_map_list=segmentation_map_list)

        # EarlyStopping
        if nr_epochs_not_improved >= patience:
            logger.my_print('No internal validation improvement during the last {} consecutive epochs. '
                            'Stop training.'.format(nr_epochs_not_improved))
            return (train_loss_values_list, val_loss_values_list, test_loss_values_list, train_mse_values_list,
                    val_mse_values_list, test_mse_values_list, train_auc_values_list, val_auc_values_list,
                    test_auc_values_list, lr_list, batch_size_list, epoch + 1, best_epoch)

        # Scheduler
        if nr_epochs_not_improved_lr >= patience_lr:
            # LR_Finder scheduler
            if scheduler_name == 'lr_finder':
                for param_group in optimizer.param_groups:
                    last_lr = param_group['lr']
                new_lr, optimizer = misc.learning_rate_finder(model=model, train_dataloader=train_dataloader,
                                                              val_dataloader=val_dataloader, optimizer=optimizer,
                                                              loss_function=loss_function, data_aug_p=data_aug_p,
                                                              data_aug_strength=data_aug_strength,
                                                              perform_augmix=perform_augmix,
                                                              mixture_width=mixture_width, mixture_depth=mixture_depth,
                                                              augmix_strength=augmix_strength,
                                                              mean=mean, std=std,
                                                              grad_max_norm=grad_max_norm, device=device, logger=logger)
                if new_lr is None:
                    # Use adjusted version of old but valid lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(last_lr * config.factor_lr, config.min_lr)

            # ReduceLROnPlateau scheduler
            elif scheduler_name == 'reduce_lr':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * config.factor_lr, config.min_lr)

            # Batch_size scheduler
            elif scheduler_name == 'batch_size':
                new_batch_size = min(cur_batch_size * 2, max_batch_size)
                train_dl_args_dict['batch_size'] = new_batch_size
                train_dataloader = dl_class(**train_dl_args_dict)

            nr_epochs_not_improved_lr = 0

        # Manual_LR scheduler: LR per epoch
        if (scheduler_name == 'manual_lr') and (len(manual_lr) > 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = manual_lr[0]
            manual_lr.pop(0)

        # Next optimizer
        if nr_epochs_not_improved_opt >= patience_opt:
            # Consider new optimizer
            if len(optimizer_name_next) >= 1:
                optimizer_name_i = optimizer_name_next[0]
                optimizer_name_next.pop(0)
                # Get current learning rate
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                optimizer = misc.get_optimizer(optimizer_name=optimizer_name_i, model=model, lr=lr, momentum=momentum,
                                               weight_decay=weight_decay, hessian_power=hessian_power,
                                               num_batches_per_epoch=num_batches_per_epoch, use_lookahead=use_lookahead,
                                               lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha, logger=logger)
                scheduler.optimizer = optimizer
                logger.my_print('Next optimizer: {}.'.format(optimizer_name_i))

            nr_epochs_not_improved_opt = 0

    return (
    train_loss_values_list, val_loss_values_list, test_loss_values_list, train_mse_values_list, val_mse_values_list,
    test_mse_values_list, train_auc_values_list, val_auc_values_list, test_auc_values_list, lr_list, batch_size_list,
    epoch + 1, best_epoch)

def optuna_objective(trial):
    # Make sure that data_preproc_config.use_umcg == True
    use_umcg = data_preproc_config.use_umcg
    assert use_umcg

    # Initialize variables
    wandb_mode = config.wandb_mode
    torch_version = config.torch_version
    seed = config.seed
    device = config.device
    cudnn_benchmark = config.cudnn_benchmark
    exp_dir = config.exp_dir
    filename_stratified_sampling_test_csv = config.filename_stratified_sampling_test_csv
    filename_stratified_sampling_full_csv = config.filename_stratified_sampling_full_csv
    perform_stratified_sampling_full = config.perform_stratified_sampling_full
    sampling_type = config.sampling_type
    cv_folds = config.cv_folds # For testing, it will be equal to one
    cv_type = config.cv_type # it should be put at 'stratified'.
    use_sampler = config.use_sampler # For uing WeightedRandomSampler (oversampling)
    perform_data_aug = config.perform_data_aug # True
    rand_cropping_size = config.rand_cropping_size # This shrinks the number of slide, here from 100 to 96
    input_size = config.input_size
    resize_mode = config.resize_mode # 'area', using for upsampling
    seg_orig_labels = config.seg_orig_labels # The name of the OARs from original form
    seg_target_labels = config.seg_target_labels # Set the important targets one, some with less important 0.1, and redundant zero.

    # Deep Learning model
    dataloader_drop_last = config.dataloader_drop_last # This ignores the last batch
    data_aug_p = config.data_aug_p # Probability of data augmentation
    data_aug_strength = config.data_aug_strength # Is set to be 1 and it means they are as powerful as the real data
    perform_augmix = config.perform_augmix
    mixture_depth = config.mixture_depth
    mixture_width = config.mixture_width
    augmix_strength = config.augmix_strength
    model_name = config.model_name # First model 'dcnn_lrelu', but it was changed here in the main program.
    features_dl = config.features_dl # This should be changed based on .csv file
    resnet_shortcut = config.resnet_shortcut
    filters = config.filters
    kernel_sizes = config.kernel_sizes
    strides = config.strides
    pad_value = config.pad_value
    lrelu_alpha = config.lrelu_alpha
    pooling_conv_filters = config.pooling_conv_filters
    perform_pooling = config.perform_pooling
    linear_units = config.linear_units
    clinical_variables_position = config.clinical_variables_position # This adds the important clinical variables to the model
    clinical_variables_linear_units = config.clinical_variables_linear_units
    clinical_variables_dropout_p = config.clinical_variables_dropout_p
    dropout_p = config.dropout_p
    use_bias = config.use_bias
    num_classes = config.num_classes # Maybe this one is the number of different categories in the label column
    num_ohe_classes = config.num_ohe_classes
    pretrained_path = config.pretrained_path # This one is a path to pretrained model, maybe is used for transferring networks.
    weight_init_name = config.weight_init_name # The following four parameters are about weight initializer. 
    kaiming_a = config.kaiming_a
    kaiming_mode = config.kaiming_mode
    kaiming_nonlinearity = config.kaiming_nonlinearity
    gain = config.gain
    optimizer_name = config.optimizer_name # This one is about the optimizer
    optimizer_name_next = config.optimizer_name_next
    loss_function_name = config.loss_function_name
    manual_lr = config.manual_lr # Learning rate properties
    base_lr = config.lr
    momentum = config.momentum # Optimizer properties
    weight_decay = config.weight_decay  # L2 regularization penalty
    hessian_power = config.hessian_power # Hessian Optimizer property
    use_lookahead = config.use_lookahead # Another Optimizer method
    lookahead_k = config.lookahead_k
    lookahead_alpha = config.lookahead_alpha
    lr_finder_num_iter = config.lr_finder_num_iter
    scheduler_name = config.scheduler_name # Learning rate scheduker, here is Cosine
    warmup_batches = config.warmup_batches
    T_0 = config.T_0 # Number of epoches befor restarting
    T_mult = config.T_mult # A factor increases after a restart
    eta_min = config.eta_min # Minimum learning rate
    step_size_up = config.step_size_up # number of itteration in each cycle
    gamma = config.gamma # Learning rate factor for each epoch
    step_size = config.step_size # Learning rate factor for each epoch
    grad_max_norm = config.grad_max_norm # (None), no grad clipping
    label_weights = config.label_weights # Weight of the classes
    label_smoothing = config.label_smoothing # (0) no label smoothing
    loss_weights = config.loss_weights # Weights of the different loss functions
    nr_of_decimals = config.nr_of_decimals # Number of decimals
    # max_epochs = config.max_epochs
    batch_size = config.batch_size # Some parameters of Training config
    max_batch_size = config.max_batch_size
    patience_lr = config.patience_lr
    patience_opt = config.patience_opt
    patience = config.patience # Limit the number of running epoches

    # Logistic regression model
    patient_id_col = data_preproc_config.patient_id_col
    features_lr = data_preproc_config.features
    endpoint = data_preproc_config.endpoint
    patient_id_length = data_preproc_config.patient_id_length

    # Initialize classes
    sigmoid_act = torch.nn.Sigmoid() # Is the activation function here
    softmax_act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=num_ohe_classes) # Determine the descrete classes
    mse_metric = MSEMetric() # mean squared error
    auc_metric = ROCAUCMetric() # AUC-ROC
    excl = ''

    # Set seed for reproducibility
    torch.manual_seed(seed=seed)
    set_determinism(seed=seed)
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark # For now, it is equal to False, but it may change to True in the future.

    # Optuna
    optuna_batch_size_list = config.optuna_batch_size_list
    optuna_filters_list = config.optuna_filters_list
    optuna_min_kernel_size = config.optuna_min_kernel_size
    optuna_max_kernel_size = config.optuna_max_kernel_size
    optuna_linear_units_size = config.optuna_linear_units_size
    optuna_clinical_variables_linear_units_size = config.optuna_clinical_variables_linear_units_size
    optuna_linear_units_list = config.optuna_linear_units_list
    optuna_clinical_variables_linear_units_list = config.optuna_clinical_variables_linear_units_list
    optuna_features_dl_list = config.optuna_features_dl_list
    T0 = config.T0

    # (Optuna) 2. Suggest values of the hyperparameters using a trial object.
    data_aug_strength = trial.suggest_float('data_aug_strength', 0.0, 3.0)
    data_aug_p = trial.suggest_float('data_aug_p', 0.0, 0.5)
    augmix_strength = trial.suggest_float('augmix_strength', 0.0, 3.0)
    batch_size_idx = trial.suggest_int('batch_size_idx', 0, len(optuna_batch_size_list) - 1, 1)
    batch_size = optuna_batch_size_list[batch_size_idx] 
    
    features_dl_idx = trial.suggest_int('features_dl_idx', 0, len(optuna_features_dl_list) - 1, 1)
    features_dl = optuna_features_dl_list[features_dl_idx] # Fix this part to get more than one input feature

    loss_function_weights_ce = trial.suggest_float('loss_function_weights_ce', 0.0, 1.0)
    loss_function_weights_f1 = trial.suggest_float('loss_function_weights_f1', 0.0, 1.0)
    loss_function_weights_dice = trial.suggest_float('loss_function_weights_dice', 0.0, 1.0)
    loss_function_weights_l1 = trial.suggest_float('loss_function_weights_l1', 0.0, 1.0)
    loss_function_weights_ranking = trial.suggest_float('loss_function_weights_ranking', 0.0, 1.0)
    loss_function_weights_softauc = trial.suggest_float('loss_function_weights_softauc', 0.0, 1.0)
    loss_weights = [loss_function_weights_ce, loss_function_weights_f1, loss_function_weights_dice, 
                                loss_function_weights_l1, loss_function_weights_ranking, loss_function_weights_softauc]

    optimizer_name = trial.suggest_categorical('optimizer_name', ['adam','rmsprop'])

    
    linear_units_size_idx = trial.suggest_int('linear_units_size_idx', 0, len(optuna_linear_units_size) -1, 1)
    linear_units_size = optuna_linear_units_size[linear_units_size_idx]
    linear_units = list()
    dropout_p = list()
    
    for _ in range(linear_units_size): 
        linear_units_idx = trial.suggest_int('linear_units_idx', 0, len(optuna_linear_units_list) -1, 1)
        linear_units_j = optuna_linear_units_list[linear_units_idx]
        linear_units.append(linear_units_j)
        dropout_p_j = trial.suggest_float('dropout_p_j', 0.0, 0.5)
        dropout_p.append(dropout_p_j)

    clinical_variables_linear_units_size_idx = trial.suggest_int('clinical_variables_linear_units_size_idx', 0, len(optuna_clinical_variables_linear_units_size) -1, 1)
    clinical_variables_linear_units_size= optuna_clinical_variables_linear_units_size[clinical_variables_linear_units_size_idx]
    clinical_variables_linear_units = list()
    clinical_variables_dropout_p = list()

    for _ in range(clinical_variables_linear_units_size): 
        clinical_variables_linear_units_idx = trial.suggest_int('clinical_variables_linear_units_idx', 0, len(optuna_clinical_variables_linear_units_list) -1, 1)
        clinical_variables_linear_units_j = optuna_clinical_variables_linear_units_list[clinical_variables_linear_units_idx]
        clinical_variables_linear_units.append(clinical_variables_linear_units_j)
        clinical_variables_dropout_p_j = trial.suggest_float('clinical_variables_dropout_p_j', 0.0, 0.5)
        clinical_variables_dropout_p.append(clinical_variables_dropout_p_j)

    clinical_variables_position = trial.suggest_int('clinical_variables_position', 0, linear_units_size - 1, 1)
    
    filter_size_idx = trial.suggest_int('filter_size_idx', 0, len(optuna_filters_list) - 1, 1)
    filter_size = optuna_filters_list[filter_size_idx]
    n_layers = trial.suggest_int('n_layers', 4, 4) # (4, 4) for resnet_lrelu #, 5)
    filters = filter_size[:n_layers]  
    kernel_sizes_i_tmp = [optuna_max_kernel_size]
    kernel_sizes = list()
    strides = list()

    for i in range(n_layers):  
        # Note: `min(kernel_sizes_i_tmp)` is used to make sure that kernel_size does not become larger for later layers
        # min(..., 128 / (2**(i+1)) is used to make sure that kernel_size is not larger than feature_map size
        if round(128 / (2**(i+1))) >= optuna_min_kernel_size:
            kernel_sizes_i = trial.suggest_int('kernel_sizes_{}'.format(i), 
                                                optuna_min_kernel_size, 
                                                min(kernel_sizes_i_tmp))
        # If the feature map size is smaller than `optuna_min_kernel_size` (e.g.,: 2x2 vs. 3), then use 
        # kernel_size smaller than `optuna_min_kernel_size`.
        # elif: round(128 / (2**(i+1))) < optuna_min_kernel_size:
        else: 
            kernel_sizes_i = trial.suggest_int('kernel_sizes_{}'.format(i), 
                                                1, min(min(kernel_sizes_i_tmp), 128 / (2**(i+1)))) 

        kernel_sizes_i_tmp.append(kernel_sizes_i)
        kernel_sizes.append([1, kernel_sizes_i, kernel_sizes_i])

        if i == n_layers - 1:
            stride_value_last_layer = trial.suggest_int('stride_value_last_layer', 1, 2) 
            strides.append([stride_value_last_layer] * 3)
        else:
            strides.append([2] * 3)


    lr = trial.suggest_float('lr', 5e-5, 1e-3)

    optuna_step_size_up = [16*batch_size, 32*batch_size, 64*batch_size]
    scheduler_name = trial.suggest_categorical('scheduler_name', ['cyclic'])
    if scheduler_name is not None:
        if scheduler_name == 'cosine':
            T_0 = trial.suggest_int('T_0', 8, 64)
        elif scheduler_name == 'cyclic':
            step_size_up_idx = trial.suggest_int('step_size_up_idx', 0, len(optuna_step_size_up) - 1, 1)
            step_size_up = optuna_step_size_up[step_size_up_idx]
        elif scheduler_name == 'exponential':
            gamma = trial.suggest_float('gamma',0.9,1.0)
        else:
            raise ValueError('Scheduler_name = {} is not available'.format(scheduler_name))
        
    use_momentum = trial.suggest_categorical('use_momentum', [True, False])
    if use_momentum:
        momentum = trial.suggest_float('momentum', 0, 1.0)
    else:
        momentum = 0
        
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)

    # (Add this one if you add features)
    # seg_target_labels = list()  
    # for k in range(17):
    #     seg_target_labels_k = trial.suggest_float('seg_target_labels_{}'.format(k), 0.0, 1.0)
    #     seg_target_labels.append(seg_target_labels_k)	
    # use_bias = trial.suggest_categorical('use_bias', [True, False])


    # Initialize W&B
    # run = wandb.init(project='DL_NTCP', reinit=True, mode=wandb_mode)
    # Initialize data objects
    train_auc_list, train_auc_lr_list = list(), list()
    val_auc_list, val_auc_lr_list = list(), list()
    train_auc_value, val_auc_value, test_auc_value = [None] * 3
    train_auc_list_mean, val_auc_list_mean = [None] * 2
    test_y_pred_sum, test_y_sum = [None] * 2
    test_y_pred_lr_sum, test_y_lr_sum = [None] * 2
    norm_mean_dict, norm_std_dict = dict(), dict()
    if model_name in ['efficientnet-b{}'.format(i) for i in range(9)] + \
            ['efficientnetv2_{}'.format(i) for i in ['s', 'm']] + \
            ['resnet_original' + 'resnext_original']:
        # Only batch_size > 1 is allowed for batch_norm
        assert batch_size > 1
        dataloader_drop_last = True

    for fold in range(cv_folds):
        globals()['ijk'] += 1
        
        # Initialize variables
        logger = initialize(torch_version=torch_version, device=device)
        logger.my_print('Model name: {}'.format(model_name))
        logger.my_print('Fold: {}'.format(fold))
        logger.my_print('Seed: {}'.format(seed))
        test_loss_value, test_mse_value, test_auc_value, test_auc_mean, test_auc_lr, test_auc_lr_mean = [None] * 6
        if pretrained_path is not None: # It means that if you have a pretrained model.
            pretrained_path_i = os.path.join(pretrained_path, str(fold), config.filename_best_model_pth)
        else:
            pretrained_path_i = None
        start = time.time()

        # Cross-Validation
        if cv_folds > 1:
            if fold == 0:
                # Fetch training, internal validation and test files (get the mentioned files)
                # Basically, it makes the train,val, and test files. Also, it is used for adding perform_test_run
                cv_dict_0, cv_dict_1, test_dict = load_data.get_files(  
                    sampling_type=sampling_type, features=features_dl,
                    filename_stratified_sampling_test_csv=filename_stratified_sampling_test_csv,
                    filename_stratified_sampling_full_csv=filename_stratified_sampling_full_csv,
                    perform_stratified_sampling_full=perform_stratified_sampling_full,
                    seed=seed, logger=logger)

                # Combine training and internal validation files
                cv_dict = cv_dict_0 + cv_dict_1

                # Perform Stratified CV
                if cv_type == 'stratified':
                    logger.my_print('Performing stratified {}-fold CV.'.format(cv_folds))
                    # Create stratified CV class
                    cv_object = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                    # Determine endpoint value to stratify on
                    cv_y = [x['label'] for x in cv_dict]
                    # For each fold: determine train and val indices
                    cv_idx_list = list()
                    # Source: https://www.programcreek.com/python/example/91147/sklearn.model_selection.StratifiedKFold
                    # Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
                    # Note that providing y is sufficient to generate the splits and hence np.zeros(len(cv_y)) may be
                    # used as a placeholder for X instead of actual training data.
                    for idx in cv_object.split(X=cv_dict, y=cv_y):
                        cv_idx_list.append(idx)

            # For this fold: select training and internal validation indices
            train_idx, valid_idx = cv_idx_list[fold]

            # Fetch training set using cv indices in 'cv_idx_list'
            train_dict = list()
            for i in train_idx:
                train_dict.append(cv_dict[i])

            # Fetch internal validation set using cv indices in 'cv_idx_list'
            val_dict = list()
            for i in valid_idx:
                val_dict.append(cv_dict[i])
            assert len(train_dict) + len(val_dict) == len(cv_dict)

        # Training-validation-test split
        else:
            train_dict, val_dict, test_dict = [None] * 3
        
        print('hiiiiiiiiiiiiiiiiiiiiiii', features_dl)
        (train_dl, val_dl, test_dl, train_ds, dl_class, train_dl_args_dict, batch_size, channels,
        depth, height, width, n_train_0, n_train_1, n_val_0, n_val_1, n_test_0, n_test_1,
        n_features, train_dict, val_dict, test_dict, norm_mean, norm_std) = \
            load_data.main(train_dict=train_dict, val_dict=val_dict, test_dict=test_dict,
                        features=features_dl, perform_data_aug=perform_data_aug, data_aug_p=data_aug_p,
                        data_aug_strength=data_aug_strength, rand_cropping_size=rand_cropping_size,
                        input_size=input_size, resize_mode=resize_mode, seg_orig_labels=seg_orig_labels,
                        seg_target_labels=seg_target_labels, batch_size=batch_size,
                        use_sampler=use_sampler, fold=fold, drop_last=dataloader_drop_last,
                        sampling_type=sampling_type,
                        filename_stratified_sampling_test_csv=filename_stratified_sampling_test_csv,
                        filename_stratified_sampling_full_csv=filename_stratified_sampling_full_csv,
                        perform_stratified_sampling_full=perform_stratified_sampling_full,
                        seed=seed, device=device, logger=logger)
        logger.my_print('Input shape: {}.'.format([batch_size, channels, depth, height, width]))
        num_batches_per_epoch = len(train_dl)
        norm_mean_dict[fold] = norm_mean
        norm_std_dict[fold] = norm_std

        # Initialize model
        print('hiiiiiiiiiiiiiiiiiiii', n_features)
        model = misc.get_model(model_name=model_name, num_ohe_classes=num_ohe_classes, channels=channels,
                            depth=depth, height=height, width=width, n_features=n_features,
                            resnet_shortcut=resnet_shortcut,
                            num_classes=num_classes, filters=filters, kernel_sizes=kernel_sizes, strides=strides,
                            pad_value=pad_value,
                            lrelu_alpha=lrelu_alpha, dropout_p=dropout_p, pooling_conv_filters=pooling_conv_filters,
                            perform_pooling=perform_pooling, linear_units=linear_units,
                            clinical_variables_position=clinical_variables_position,
                            clinical_variables_linear_units=clinical_variables_linear_units,
                            clinical_variables_dropout_p=clinical_variables_dropout_p,
                            use_bias=use_bias, pretrained_path=pretrained_path_i, logger=logger)
        model.to(device=device)

        # Compile model (PyTorch 2) to get rid of
        # if torch_version.startswith('2.'):
        #     model = torch.compile(model)

        # Weight initialization
        if 'selu' in model_name:
            # Source: https://github.com/bioinf-jku/SNNs/tree/master/Pytorch
            weight_init_name = 'kaiming_normal'
            kaiming_a = None
            kaiming_mode = 'fan_in'
            kaiming_nonlinearity = 'linear'

        logger.my_print('Weight init name: {}.'.format(weight_init_name))

        if pretrained_path_i is not None:
            logger.my_print('Using pretrained weights: {}.'.format(pretrained_path_i))

        elif ((pretrained_path_i is None) and
            (weight_init_name is not None) and
            ('efficientnet' not in model_name) and
            ('convnext' not in model_name) and
            ('resnext' not in model_name)):

            # Source: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
            model.apply(lambda m: misc.weights_init(m=m,
                                                    weight_init_name=weight_init_name, 
                                                    kaiming_a=kaiming_a,
                                                    kaiming_mode=kaiming_mode,
                                                    kaiming_nonlinearity=kaiming_nonlinearity, 
                                                    gain=gain,
                                                    logger=logger))

            logger.my_print('Using our own weights init scheme for {}.'.format(model_name))
        else:
            logger.my_print('Using default PyTorch weights init scheme for {}.'.format(model_name))

        # Create and save model summary
        # summary() only accepts non-zero integers, i.e. n_features = 0 is invalid in `input_size`.
        # However n_features = 0 in get_model() will make `n_features` in summary() redundant, so
        # we just use `1` in the input of summary().
        # Important note: torchinfo.summary() somehow uses some random() on input_size, because it will create
        # random input. I.e. input_size changes the next random values.
        total_params = misc.get_model_summary(model=model, 
                                            input_size=[(batch_size, channels, depth, height, width),
                                                                    (batch_size, max(n_features, 1))],
                                            device=device,
                                            logger=logger)

        # Initialize loss function
        loss_function = misc.get_loss_function(loss_function_name=loss_function_name, 
                                            label_weights=label_weights,
                                            label_smoothing=label_smoothing, 
                                            loss_weights=loss_weights,
                                            device=device)

        # Compile model (PyTorch 2) #########################
        # if torch_version.startswith('2.') and not os.name == 'nt':
        #     model = torch.compile(model)

        # Collect model's gradients and topology
        # wandb.watch(model, loss_function, log_freq=1, log='all')

        # Initialize optimizer
        if scheduler_name == 'manual_lr':
            lr = manual_lr[0]
            manual_lr.pop(0)

        else:
            lr = base_lr

        optimizer = misc.get_optimizer(optimizer_name=optimizer_name, model=model, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay, hessian_power=hessian_power,
                                    num_batches_per_epoch=num_batches_per_epoch, use_lookahead=use_lookahead,
                                    lookahead_k=lookahead_k,
                                    lookahead_alpha=lookahead_alpha, logger=logger)

        # Perform learning rate finder
        if lr_finder_num_iter > 0 and (config.max_epochs > 0):
            lr, optimizer = misc.learning_rate_finder(
                model=model, train_dataloader=train_dl, val_dataloader=val_dl, optimizer=optimizer,
                optimizer_name=optimizer_name, loss_function=loss_function, data_aug_p=data_aug_p,
                data_aug_strength=data_aug_strength, perform_augmix=perform_augmix, mixture_width=mixture_width,
                mixture_depth=mixture_depth, augmix_strength=augmix_strength, mean=norm_mean_dict[fold],
                std=norm_std_dict[fold], grad_max_norm=grad_max_norm, lr_finder_num_iter=lr_finder_num_iter,
                save_filename=config.filename_lr_finder_png, device=device, logger=logger)
        starting_lr = lr

        # If lr = None (i.e. lr cannot be found), then skip iteration of the for-loop
        if lr is None:
            # continue
            max_epochs = 0
        else:
            max_epochs = config.max_epochs

        # Initialize scheduler
        # Note: should be performed after learning_rate_finder()
        scheduler = misc.get_scheduler(scheduler_name=scheduler_name, optimizer=optimizer, optimal_lr=lr, T_0=T_0,
                                    T_mult=T_mult, eta_min=eta_min, step_size_up=step_size_up, gamma=gamma,
                                    step_size=step_size, warmup_batches=warmup_batches,
                                    num_batches_per_epoch=num_batches_per_epoch,
                                    lr_finder_num_iter=lr_finder_num_iter, logger=logger)

        # Train model
        (train_loss_values, val_loss_values, test_loss_values, train_mse_values, val_mse_values, test_mse_values,
        train_auc_values, val_auc_values, test_auc_values, lr_values, batch_size_values, epochs, best_epoch) = \
            train(model=model, train_dataloader=train_dl, val_dataloader=val_dl, test_dataloader=test_dl,
                    mean=norm_mean_dict[fold], std=norm_std_dict[fold],
                    optimizer=optimizer, optimizer_name_next=optimizer_name_next,
                    dl_class=dl_class, train_dl_args_dict=train_dl_args_dict, num_classes=num_classes,
                    max_epochs=max_epochs, perform_augmix=perform_augmix, mixture_width=mixture_width,
                    mixture_depth=mixture_depth, augmix_strength=augmix_strength, optimizer_name=optimizer_name,
                    grad_max_norm=grad_max_norm, scheduler_name=scheduler_name, scheduler=scheduler,
                    loss_function_name=loss_function_name, sigmoid_act=sigmoid_act, softmax_act=softmax_act,
                    to_onehot=to_onehot, patience=patience, patience_lr=patience_lr, patience_opt=patience_opt,
                    loss_function=loss_function, auc_metric=auc_metric, mse_metric=mse_metric,
                    data_aug_p=data_aug_p, data_aug_strength=data_aug_strength, max_batch_size=max_batch_size,
                    manual_lr=manual_lr, momentum=momentum, weight_decay=weight_decay, hessian_power=hessian_power,
                    num_batches_per_epoch=num_batches_per_epoch, use_lookahead=use_lookahead,
                    lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha, device=device,
                    model_name=model_name, exp_dir=exp_dir,
                    logger=logger)

        if max_epochs > 0:
            # Plot training and internal validation losses
            y_list = [[train_loss_values, val_loss_values],
                    [train_mse_values, val_mse_values],
                    [train_auc_values, val_auc_values], [lr_values],
                    [batch_size_values]]
            y_label_list = ['Loss', 'MSE', 'AUC', 'LR', 'Batch_size']
            legend_list = [['Training', 'Internal validation']] * (len(y_list) - 2) + [None, None]
            misc.plot_values(y_list=y_list, y_label_list=y_label_list, best_epoch=best_epoch, legend_list=legend_list,
                            figsize=config.figsize, save_filename=os.path.join(exp_dir, config.filename_results_png))

            # Load best model
            model.load_state_dict(torch.load(os.path.join(exp_dir, config.filename_best_model_pth)))

            # Store predictions on training and validation set
            # IMPORTANT NOTE: train_dl still performs data augmentation!
            train_loss_value, train_mse_value, train_auc_value, train_patient_ids, train_y_pred_list, train_y_list = validate(
                        model=model, dataloader=train_dl, mean=norm_mean_dict[fold], std=norm_std_dict[fold],
                        mode='test', 
                        loss_function=loss_function, loss_function_name=loss_function_name,
                        sigmoid_act=sigmoid_act, softmax_act=softmax_act, auc_metric=auc_metric, mse_metric=mse_metric,
                        to_onehot=to_onehot, num_classes=num_classes, model_name=model_name, exp_dir=exp_dir, device=device,
                        logger=logger, save_outputs=False)

            val_loss_value, val_mse_value, val_auc_value, val_patient_ids, val_y_pred_list, val_y_list = validate(
                    model=model, dataloader=val_dl, mean=norm_mean_dict[fold], std=norm_std_dict[fold],
                    loss_function=loss_function, loss_function_name=loss_function_name,
                    sigmoid_act=sigmoid_act, softmax_act=softmax_act, auc_metric=auc_metric, mse_metric=mse_metric,
                    to_onehot=to_onehot, num_classes=num_classes, model_name=model_name, exp_dir=exp_dir,
                    device=device,
                    mode='test', logger=logger, save_outputs=False)

            all_patient_ids = train_patient_ids + val_patient_ids
            all_y_pred_list = train_y_pred_list + val_y_pred_list
            all_y_list = train_y_list + val_y_list
            all_modes = ['train'] * len(train_patient_ids) + ['val'] * len(val_patient_ids)

            train_auc_list.append(train_auc_value)
            val_auc_list.append(val_auc_value)

            # Test evaluate model
            if test_dl is not None:
                test_loss_value, test_mse_value, test_auc_value, test_patient_ids, test_y_pred_list, test_y_list = validate(
                    model=model, dataloader=test_dl, mean=norm_mean_dict[fold], std=norm_std_dict[fold],
                    mode='test', logger=logger,
                    loss_function=loss_function, loss_function_name=loss_function_name,
                    sigmoid_act=sigmoid_act, softmax_act=softmax_act, auc_metric=auc_metric, mse_metric=mse_metric,
                    to_onehot=to_onehot, num_classes=num_classes, model_name=model_name, exp_dir=exp_dir,
                    device=device,
                    save_outputs=True
                    )

                # Store predictions on test set
                all_patient_ids += test_patient_ids
                all_y_pred_list += test_y_pred_list
                all_y_list += test_y_list
                all_modes += ['test'] * len(test_patient_ids)

                if (test_y_pred_sum is None) and (test_y_sum is None):
                    test_y_pred_sum = test_y_pred_list
                    test_y_sum = test_y_list
                else:
                    test_y_pred_sum = [x + y for x, y in zip(test_y_pred_sum, test_y_pred_list)]
                    test_y_sum = [x + y for x, y in zip(test_y_sum, test_y_list)]

            # Save predictions on training, validation and test set
            misc.save_predictions(patient_ids=all_patient_ids, y_pred_list=all_y_pred_list, y_true_list=all_y_list,
                                mode_list=all_modes, num_classes=num_classes, model_name=model_name + '_all',
                                exp_dir=exp_dir, logger=logger)

        # Run logistic regression (LR) model (reference model)
        features_csv = os.path.join(config.data_dir, data_preproc_config.filename_features_csv)
        patient_ids_json = os.path.join(exp_dir, config.filename_train_val_test_patient_ids_json.format(fold))

        (train_patient_ids_lr, train_y_pred_lr, train_y_lr, val_patient_ids_lr, val_y_pred_lr, val_y_lr,
        test_patient_ids_lr, test_y_pred_lr, test_y_lr, lr_coefficients) = \
            run_logistic_regression(features_csv=features_csv, patient_id_col=patient_id_col,
                                    baseline_col=data_preproc_config.baseline_col,
                                    submodels_features=data_preproc_config.submodels_features, features=features_lr,
                                    lr_coefficients=data_preproc_config.lr_coefficients, endpoint=endpoint,
                                    patient_id_length=patient_id_length, patient_ids_json=patient_ids_json, seed=seed,
                                    nr_of_decimals=nr_of_decimals, logger=logger)

        # Compute training and internal validation AUC of LR model
        train_y_pred_lr_list = [x for x in train_y_pred_lr]
        train_y_lr_list = [to_onehot(i) for i in train_y_lr]
        train_auc_lr = misc.compute_auc(y_pred_list=train_y_pred_lr_list, y_true_list=train_y_lr_list,
                                        auc_metric=auc_metric)
        train_auc_lr_list.append(train_auc_lr)

        val_y_pred_lr_list = [x for x in val_y_pred_lr]
        val_y_lr_list = [to_onehot(i) for i in val_y_lr]
        val_auc_lr = misc.compute_auc(y_pred_list=val_y_pred_lr_list, y_true_list=val_y_lr_list, auc_metric=auc_metric)
        val_auc_lr_list.append(val_auc_lr)

        # Store predictions on training and validation set of LR model
        # IMPORTANT NOTE: train_dl still performs data augmentation!
        all_patient_ids_lr = train_patient_ids_lr + val_patient_ids_lr
        all_y_pred_lr_list = train_y_pred_lr_list + val_y_pred_lr_list
        all_y_lr_list = train_y_lr_list + val_y_lr_list
        all_modes_lr = ['train'] * len(train_patient_ids_lr) + ['val'] * len(val_patient_ids_lr)

        # Test evaluate (LR model)
        if test_dl is not None:
            test_y_pred_lr_list = [x for x in test_y_pred_lr]
            test_y_lr_list = [to_onehot(i) for i in test_y_lr]
            test_auc_lr = misc.compute_auc(y_pred_list=test_y_pred_lr_list, y_true_list=test_y_lr_list,
                                        auc_metric=auc_metric)

            # Determine patient_ids in test set (for misc.save_predictions())
            mode_lr_list = ['test'] * len(test_patient_ids_lr)

            all_patient_ids_lr += test_patient_ids_lr
            all_y_pred_lr_list += test_y_pred_lr_list
            all_y_lr_list += test_y_lr_list
            all_modes_lr += mode_lr_list

            # Save outputs to csv
            misc.save_predictions(patient_ids=test_patient_ids_lr, y_pred_list=test_y_pred_lr_list,
                                y_true_list=test_y_lr_list, mode_list=mode_lr_list, num_classes=num_classes,
                                model_name='lr', exp_dir=exp_dir, logger=logger)

            if (test_y_pred_lr_sum is None) and (test_y_lr_sum is None):
                test_y_pred_lr_sum = test_y_pred_lr_list
                test_y_lr_sum = test_y_lr_list
            else:
                test_y_pred_lr = [x for x in test_y_pred_lr_list]
                test_y_pred_lr_sum = [x + y for x, y in zip(test_y_pred_lr_sum, test_y_pred_lr_list)]
                test_y_lr_sum = [x + y for x, y in zip(test_y_lr_sum, test_y_lr_list)]

            # Ensemble (DL and LR model)
            if (fold == cv_folds - 1) and (cv_folds > 1):
                # Make sure that test_y_sum and test_y_lr_sum have only `num_classes` different tensors:
                # e.g. cv_folds = 5: {(0.0, 5.0), (5.0, 0.0)}
                if max_epochs > 0:
                    # print('HIIIIIIIIIIIIIIII', len(set([tuple(x.tolist()) for x in test_y_sum])), num_classes)
                    # print('HIIIIIIIIIIIIIIII', test_y_sum, test_y_lr_sum)
                    # assert len(set([tuple(x.tolist()) for x in test_y_sum])) == num_classes
                    test_y_pred_ens = [x / cv_folds for x in test_y_pred_sum]
                    test_y_ens = [x / cv_folds for x in test_y_sum]
                    misc.save_predictions(patient_ids=test_patient_ids, y_pred_list=test_y_pred_ens,
                                        y_true_list=test_y_ens, mode_list=mode_lr_list, num_classes=num_classes,
                                        model_name=model_name + '_ens', exp_dir=exp_dir, logger=logger)
                # print('hiiiiiiiiiiiiiiii', test_y_lr_sum, num_classes)
                assert len(set([tuple(x.tolist()) for x in test_y_lr_sum])) == num_classes
                test_y_pred_lr_ens = [x / cv_folds for x in test_y_pred_lr_sum]
                test_y_lr_ens = [x / cv_folds for x in test_y_lr_sum]
                misc.save_predictions(patient_ids=test_patient_ids_lr, y_pred_list=test_y_pred_lr_ens,
                                    y_true_list=test_y_lr_ens, mode_list=mode_lr_list, num_classes=num_classes,
                                    model_name='lr_ens', exp_dir=exp_dir, logger=logger)

        # Save predictions on training, validation and test set
        misc.save_predictions(patient_ids=all_patient_ids_lr, y_pred_list=all_y_pred_lr_list, y_true_list=all_y_lr_list,
                            mode_list=all_modes_lr, num_classes=num_classes, model_name='lr_all', exp_dir=exp_dir,
                            logger=logger)

        if max_epochs > 0:
            # Create results.csv
            misc.create_results(train_loss=train_loss_value, val_loss=val_loss_value, test_loss=test_loss_value,
                                train_auc=train_auc_value, train_auc_lr=train_auc_lr, val_auc=val_auc_value,
                                val_auc_lr=val_auc_lr, test_auc=test_auc_value, test_auc_lr=test_auc_lr,
                                seed=seed, epochs=epochs, best_epoch=best_epoch, batch_size=batch_size,
                                starting_lr=starting_lr, patience=patience, model_name=model_name,
                                features_dl=features_dl, n_features=n_features, filters=filters,
                                kernel_sizes=kernel_sizes, strides=strides, lrelu_alpha=lrelu_alpha,
                                pooling_conv_filters=pooling_conv_filters, linear_units=linear_units,
                                dropout_p=dropout_p, use_bias=use_bias, total_params=total_params,
                                loss_weights=loss_weights,
                                n=n_train_0 + n_train_1 + n_val_0 + n_val_1 + n_test_0 + n_test_1,
                                n_train=[n_train_0, n_train_1], n_val=[n_val_0, n_val_1], n_test=[n_test_0, n_test_1],
                                train_val_test_frac=[config.train_frac, config.val_frac,
                                                    1 - config.train_frac - config.val_frac],
                                cv_folds=cv_folds, input_shape=[batch_size, channels, depth, height, width],
                                normalization_mean_dict=norm_mean_dict, normalization_std_dict=norm_std_dict,
                                ct_b_min_max=[config.ct_b_min, config.ct_b_max],
                                rtdose_b_min_max=[config.rtdose_b_min, config.rtdose_b_max],
                                seg_orig_labels=seg_orig_labels, seg_target_labels=seg_target_labels,
                                interpolation_mode_3d=[config.ct_interpol_mode_3d, config.rtdose_interpol_mode_3d,
                                                    config.segmentation_interpol_mode_3d],
                                interpolation_mode_2d=[config.ct_interpol_mode_2d, config.rtdose_interpol_mode_2d,
                                                    config.segmentation_interpol_mode_2d],
                                perform_augmix=perform_augmix, weight_init_name=weight_init_name,
                                optimizer=optimizer_name, optimizer_name_next=optimizer_name_next,
                                loss_function=loss_function_name, label_weights=label_weights,
                                label_smoothing=label_smoothing, lr_finder_num_iter=lr_finder_num_iter,
                                scheduler=scheduler_name, grad_max_norm=grad_max_norm, lr_coefficients=lr_coefficients)

        end = time.time()
        logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, nr_of_decimals)))
        logger.my_print('DONE!')
        logger.close()

        # Rename folder
        src_folder_name = os.path.join(exp_dir)
        if (fold == cv_folds - 1) and (cv_folds > 1):
            dst_folder_name = os.path.join(exp_dir + '_{}'.format(globals()['ijk']) +
                                            '_{}'.format(fold) +
                                            '_{}'.format(num_classes) +
                                            '_{}'.format(best_epoch) +
                                            '_{}'.format(model_name) +
                                            '_auc' +
                                            '_tr_{}'.format(train_auc_value) +
                                            # '_lr_{}'.format(train_auc_lr) +
                                            '_val_{}'.format(val_auc_value)
                                            # '_lr_{}'.format(val_auc_lr)
                                            )
        else:
            dst_folder_name = os.path.join(exp_dir + '_{}'.format(globals()['ijk']) + '_{}'.format(seed) +
                                            '_bsl_{}'.format(len(features_dl)) +
                                            excl + 
                                            '_{}'.format(fold) +
                                            '_{}'.format(best_epoch) +
                                            '_{}'.format(model_name) +
                                            '_params_{}'.format(total_params) +
                                            '_auc' +
                                            '_tr_{}'.format(train_auc_value) +
                                            # '_lr_{}'.format(train_auc_lr) +
                                            '_val_{}'.format(val_auc_value)
                                            # '_lr_{}'.format(val_auc_lr)
                                            )

        if test_dl is not None:
            dst_folder_name += '_test_{}'.format(test_auc_value)
            dst_folder_name += '_lr_{}'.format(test_auc_lr)

        # Results of all folds (avg and ensemble)
        if (fold == cv_folds - 1) and (cv_folds > 1):
            dst_folder_name += '_avg'

            if max_epochs > 0:
                # DL training (avg)
                train_auc_list = [x for x in train_auc_list if x is not None]
                train_auc_list_mean = round(sum(train_auc_list) / len(train_auc_list),
                                            nr_of_decimals)

            # LR training (avg)
            train_auc_lr_list = [x for x in train_auc_lr_list if x is not None]
            train_auc_lr_list_mean = round(sum(train_auc_lr_list) / len(train_auc_lr_list),
                                        nr_of_decimals)
            dst_folder_name += '_tr_{}_lr_{}'.format(train_auc_list_mean, train_auc_lr_list_mean)

            # Internal validation (avg) and test (ensemble)
            # DL model
            dst_folder_name += '_val'

            if max_epochs > 0:
                # Validation (avg)
                val_auc_list = [x for x in val_auc_list if x is not None]
                val_auc_list_mean = round(sum(val_auc_list) / len(val_auc_list), nr_of_decimals)

                # Test (ensemble)
                test_auc_mean = misc.compute_auc(y_pred_list=test_y_pred_ens,
                                                y_true_list=test_y_ens,
                                                auc_metric=auc_metric)
            dst_folder_name += '_{}'.format(val_auc_list_mean)

            # LR model
            # Validation (avg)
            val_auc_lr_list = [x for x in val_auc_lr_list if x is not None]
            val_auc_lr_list_mean = round(sum(val_auc_lr_list) / len(val_auc_lr_list),
                                        nr_of_decimals)

            dst_folder_name += '_lr_{}'.format(val_auc_lr_list_mean)
    
            # Test (ensemble)
            test_auc_lr_mean = misc.compute_auc(y_pred_list=test_y_pred_lr_ens,
                                                y_true_list=test_y_lr_ens,
                                                auc_metric=auc_metric)

            # Ensemble
            dst_folder_name += '_ens_{}'.format(test_auc_mean)
            dst_folder_name += '_lr_{}'.format(test_auc_lr_mean)

        shutil.move(src_folder_name, dst_folder_name)

        # W&B
        # run.name = dst_folder_name.replace(os.path.join(config.exp_root_dir), '')[1:]
        # # run.id = dst_folder_name.replace(os.path.join(config.exp_root_dir), '')[1:]
        # run.finish()

    # Save optuna objects
    joblib.dump(optuna_study, optuna_out_file_study)
    # Save the sampler for reproducibility after resuming study
    joblib.dump(optuna_study.sampler, optuna_out_file_sampler)
    
    # Optuna objectives to maximize, e.g., validation value
    return val_auc_list_mean


if __name__ == '__main__':
    global ijk
    globals()['ijk'] = 0

    # (Optuna) 3. Create a study object and optimize the objective function.
    # Resume study if study and sampler files exist
    # Source: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
    exp_root_dir = config.exp_root_dir
    optuna_path_pickles = config.optuna_path_pickles
    optuna_path_figures = config.optuna_path_figures
    for p in [exp_root_dir, optuna_path_pickles, optuna_path_figures]:
        create_folder_if_not_exists(p)
        
    optuna_study_name = 'optuna_study.pkl'
    optuna_sampler_name = 'optuna_sampler.pkl'
    optuna_n_trials = 100 # It should change to 100
    
    optuna_file_study_list = [x for x in os.listdir(optuna_path_pickles) if optuna_study_name in x]
    optuna_file_sampler_list = [x for x in os.listdir(optuna_path_pickles) if optuna_sampler_name in x]
    if len(optuna_file_study_list) > 0 and len(optuna_file_sampler_list) > 0:
        # Find latest study, and add 1 for the next study run
        optuna_study_run_nr = max([int(x.split('_')[0]) for x in optuna_file_study_list]) + 1
        optuna_sampler_run_nr = max([int(x.split('_')[0]) for x in optuna_file_sampler_list]) + 1
        assert optuna_study_run_nr == optuna_sampler_run_nr
        
        optuna_in_file_study = os.path.join(optuna_path_pickles, '{}_'.format(optuna_study_run_nr - 1) + optuna_study_name)
        optuna_in_file_sampler = os.path.join(optuna_path_pickles, '{}_'.format(optuna_sampler_run_nr - 1) + optuna_sampler_name)
        print('Resuming previous study: {}'.format(optuna_in_file_study))
    
        optuna_sampler = joblib.load(optuna_in_file_sampler)
        optuna_study = joblib.load(optuna_in_file_study)
    else:
        optuna_study_run_nr = 0
        optuna_sampler_run_nr = 0
        
        optuna_sampler = optuna.samplers.TPESampler(seed=123)
        optuna_study = optuna.create_study(sampler=optuna_sampler, direction='maximize')
        
    # Create study
    optuna_out_file_study = os.path.join(optuna_path_pickles, '{}_'.format(optuna_study_run_nr) + optuna_study_name)
    optuna_out_file_sampler = os.path.join(optuna_path_pickles, '{}_'.format(optuna_sampler_run_nr) + optuna_sampler_name)
    joblib.dump(optuna_study, optuna_out_file_study)
    joblib.dump(optuna_study.sampler, optuna_out_file_sampler)
    
    # Run hyperparameter tuning
    optuna_study.optimize(optuna_objective, n_trials=optuna_n_trials)
    joblib.dump(optuna_study, optuna_out_file_study)
    joblib.dump(optuna_study.sampler, optuna_out_file_sampler)
