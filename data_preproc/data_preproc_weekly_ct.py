"""
Important Tips:
    - This code is written to be used beside the main program, so I will not change the main code.
      Instead, I will add one (or two) module(s) to be used if one wants to use weeklyCTs (each of them
      does not matter) instead of masks (like me :) ).
    - There is NO resampling step here. The reason is that weeklyCTs were registered first based on
      baseline CTs; consequently, there is no need for resampling again.
    - WeeklyCTs will be cropped based on baseline segmentation bounding box. Again, the reasons are
        1. WeeklyCTs are registered base on the baseline CT. So, these CTs are limited in the bounding
           box of baseline CTs.
        2. Most of the times OARs (Organs at Risk) are smaller after RT due tou the shrinkage power of
           radiation.
      TODO: determine the bounding box base on the largest contour (first compare the baselineand weekly
            CT contours and choos the larger one). This will help to cover all the OARs' boundaries in 
            both CTs.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import pydicom as pdcm
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from pydicom import dcmread

import data_preproc_config as cfg
from data_preproc import get_single_channel_array
from data_preproc_ct_rtdose import load_ct
from data_preproc_functions import create_folder_if_not_exists, copy_file, Logger, set_default, sort_human

def get_weeklyct_path(path, weeklyct_folder_name, logger):
    """
    Write documentation
    """
    for r, d, f in os.walk(path):
        # Make a list from all the directories 
        subfolders = [os.path.join(r, folder) for folder in d]

        # Check if subfolders list is not empty before accessing its first element
        for sub in subfolders:
            
            # If the sub directory contains the name of the folder weeklyCTs were saved in.
            if weeklyct_folder_name in sub:

                try:
                    files = os.listdir(sub)
                    # Check if there are any files and if the first file contains 'dcm'
                    if files and all('dcm' in file.lower() for file in files):
                        return sub

                except IndexError:
                    logger.my_print(f'No subfolders found in {sub}', level='warning')                
    
def save_ct_arr():
    """
    Write documentation
    """    
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length
    save_root_dir = cfg.save_root_dir

    # WeeklyCT initial variables
    weeklyct_dir = cfg.weeklyct_dir
    save_dir_weeklyct = cfg.save_dir_weeklyct
    weeklyct_folder_name = cfg.weeklyct_folder_name
    filename_weeklyct_metadata_json = cfg.filename_weeklyct_metadata_json
    filename_weeklyct_npy = cfg.filename_weeklyct_npy

    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_rtdose_logging_txt))
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_weeklyct)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # Load file from data_folder_types.py
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get all patient_ids
        logger.my_print('Listing all patient ids and their files...')
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        for patient_id in tqdm(patients_list):

            logger.my_print('Patient_id: {}'.format(patient_id))
            main_path = os.path.join(weeklyct_dir, patient_id)

            logger.my_print('Patient_id: {}, Finding a path (WeeklyCTs)...'.format(patient_id))
            patient_weeklyct_path = get_weeklyct_path(main_path, weeklyct_folder_name, logger)

            # Check
            assert len(patient_weeklyct_path) > 0 

            # Load WeeklyCTs
            logger.my_print('\tLoading Pydicom Images (WeeklyCT)...')
            image_weeklyct, _ = load_ct(path_ct=patient_weeklyct_path, logger=logger)
            arr_weeklyct = sitk.GetArrayFromImage(image_weeklyct) # Extract the numpy array

            # Extracting Meta-data
            logger.my_print('\tExtracting meta-data (WeeklyCT)...')

            # Meta Data
            ds_weeklyct_dict = dict()
            ds_weeklyct_dict['ID'] = patient_id
            ds_weeklyct_dict['Path_ct'] = patient_weeklyct_path
            ds_weeklyct_dict['Direction'] = image_weeklyct.GetDirection()
            ds_weeklyct_dict['Origin'] = image_weeklyct.GetOrigin()
            ds_weeklyct_dict['Size'] = image_weeklyct.GetSize()
            ds_weeklyct_dict['Spacing'] = image_weeklyct.GetSpacing()

            ########
            # TIP: I will not resample the weeklyCTs here since I registered them before.
            # TODO: Maybe add resampling weeklyCTs to be comparable with baseline CTs.   
            ########

            # Save meta-data and numpy array
            logger.my_print('\tSaving meta-data and arrays (WeeklyCT)...')
            save_path_weeklyct = os.path.join(save_dir_weeklyct, patient_id)
            create_folder_if_not_exists(save_path_weeklyct)

            # Save meta-data dictionary as JSON
            # 'w': overwrites the file if the file exists. If the file does not exist, creates a new file for writing.
            # Source: https://tutorial.eyehunts.com/python/python-file-modes-open-write-append-r-r-w-w-x-etc/
            with open(os.path.join(save_path_weeklyct, filename_weeklyct_metadata_json), 'w') as file:
                json.dump(ds_weeklyct_dict, file, default=set_default)

            # Save as Numpy array
            np.save(file=os.path.join(save_path_weeklyct, filename_weeklyct_npy), arr=arr_weeklyct)

    end = time.time()
    logger.my_print('Elapsed time: {} seconds'.format(round(end - start, 3)))
    logger.my_print('DONE!')            

def preprocess_weeklyct_array():
    """
    To avoid confusion for the users, this function will perform almost the same
    operation as main_array() function did on the Baseline CTs' arrays.
    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length
    save_root_dir = cfg.save_root_dir
    save_dir_ct = cfg.save_dir_ct
    save_dir_dataset_full = cfg.save_dir_dataset_full

    # Initialize WeeklyCT variables
    save_dir_weeklyct = cfg.save_dir_weeklyct
    filename_weeklyct_metadata_json = cfg.filename_weeklyct_metadata_json
    filename_weeklyct_npy = cfg.filename_weeklyct_npy

    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_array_logging_txt))
    bb_size = cfg.bb_size
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_dataset_full)

    for _, d_i in zip(mode_list, data_dir_mode_list):

        # Check whether we have the same number of patients for weekly and Baseline CTs.
        patients_list_ct = os.listdir(save_dir_ct)
        patients_list_weeklyct = os.listdir(save_dir_weeklyct)

        assert patients_list_ct == patients_list_weeklyct

        patients_list = patients_list_ct
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # Load cropping regions of all patients
        if cfg.perform_cropping:
            cropping_regions = pd.read_csv(os.path.join(save_root_dir, cfg.filename_cropping_regions_csv),
                                           sep=';', index_col=0)
            # Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int, because of Excel/csv file)
            # to patient_id = '0111766' (type=str)
            cropping_regions_index_name = cropping_regions.index.name
            cropping_regions.index = ['%0.{}d'.format(patient_id_length) % x for x in cropping_regions.index]
            cropping_regions.index.name = cropping_regions_index_name

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {id}'.format(id=patient_id))

            # Load ct_metadata, for performing spacing_correction
            weeklyct_metadata = None
            if cfg.perform_spacing_correction:
                weeklyct_metadata_path = os.path.join(save_dir_weeklyct, patient_id, filename_weeklyct_metadata_json)
                json_file = open(weeklyct_metadata_path)
                weeklyct_metadata = json.load(json_file)
                logger.my_print('\tct_metadata["Spacing"][::-1] (input): {}'.format(weeklyct_metadata["Spacing"][::-1]))
                logger.my_print('\tcfg.spacing[::-1] (output): {}'.format(cfg.spacing[::-1]))

            # Extract cropping region
            cropping_region_i = None
            if cfg.perform_cropping:
                cropping_region_i = cropping_regions.loc[patient_id].to_dict()

            # Load and preprocess WeeklyCT
            logger.my_print('\t----- WeeklyCT -----')
            weeklyct_arr_path = os.path.join(save_dir_weeklyct, patient_id, filename_weeklyct_npy)
            weeklyct_arr = get_single_channel_array(arr_path=weeklyct_arr_path, dtype=None, metadata=weeklyct_metadata, is_label=False,
                                              cropping_region=cropping_region_i,
                                              bb_size=bb_size,
                                              perform_spacing_correction=cfg.perform_spacing_correction,
                                              perform_cropping=cfg.perform_cropping,
                                              perform_clipping=cfg.perform_clipping,
                                              perform_transformation=cfg.perform_transformation, logger=logger)

            # Save as Numpy array
            save_dir_dataset_full_i = os.path.join(save_dir_dataset_full, patient_id)
            create_folder_if_not_exists(save_dir_dataset_full_i)
            np.save(file=os.path.join(save_dir_dataset_full_i, 'weeklyct.npy'), arr=weeklyct_arr)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')

def preprocess_weeklyct_dataset():
    """
    This function is similar to data_preproc.main_dataset(), and it adds weeklyCTs
    to their right label folder.
    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    save_dir_dataset_full = cfg.save_dir_dataset_full

    save_dir_dataset = cfg.save_dir_dataset

    patient_id_col = cfg.patient_id_col
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_features_logging_txt))
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_dataset)

    for _, d_i in zip(mode_list, data_dir_mode_list):

        # Load Excel file containing patient_id and their endpoints/labels/targets
        endpoints_csv = os.path.join(save_root_dir, cfg.filename_endpoints_csv)
        df = pd.read_csv(endpoints_csv, sep=';')

        # Check whether the file is seperated with ','
        if len(df.columns) == 1:
            df = pd.read_csv(endpoints_csv, sep=',')

        # Create dictionary for the labels 
        labels_patients_dict = dict()
        # Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int,
        # because of Excel/csv file) to patient_id = '0111766' (type=str)
        labels_patients_dict['0'] = ['%0.{}d'.format(patient_id_length) % x for x in
                                     df[df[cfg.endpoint] == 0][patient_id_col].tolist()]
        labels_patients_dict['1'] = ['%0.{}d'.format(patient_id_length) % x for x in
                                     df[df[cfg.endpoint] == 1][patient_id_col].tolist()]

        for label, patients_list in labels_patients_dict.items():
            save_dir_dataset_label = os.path.join(save_dir_dataset, str(label))
            create_folder_if_not_exists(save_dir_dataset_label)

            # Testing for a small number of patients
            if cfg.test_patients_list is not None:
                test_patients_list = cfg.test_patients_list
                patients_list = [x for x in test_patients_list if x in patients_list]        

            for patient_id in tqdm(patients_list):
                logger.my_print('Patient_id: {id}'.format(id=patient_id))

                # Copy WeeklyCT file from dataset_full to dataset/0 and dataset/1
                src = os.path.join(save_dir_dataset_full, patient_id, 'weeklyct.npy')
                dst = os.path.join(save_dir_dataset_label, patient_id, 'weeklyct.npy')
                copy_file(src, dst)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')            

def calculate_ct_substraction():
    save_dir_dataset = cfg.save_dir_dataset
    first_array_sub = cfg.first_array_sub
    second_array_sub = cfg.second_array_sub
    filename_subtractionct = cfg.filename_subtractionct

    for r, d, f in os.walk(save_dir_dataset):
        # make a list from all the directories 
        subfolders = [os.path.join(r, folder) for folder in d]

        try:
            for subf in subfolders:
                dir_list = os.listdir(subf)
                
                if 'np' in dir_list[0].lower():
                    first_npy = np.load(os.path.join(subf, first_array_sub))
                    second_npy = np.load(os.path.join(subf, second_array_sub))
                    subtraction_npy = first_npy - second_npy
                    np.save(file=os.path.join(subf, filename_subtractionct), arr=subtraction_npy)
        
        except Exception as e:
            print(e)
            pass        

def main():
    save_ct_arr()
    preprocess_weeklyct_array()
    preprocess_weeklyct_dataset()
    # calculate_ct_substraction() Is used if wants the subtraction ct

if __name__ == '__main__':
    main()