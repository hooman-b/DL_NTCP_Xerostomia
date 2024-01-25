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
from data_preproc_ct_rtdose import load_ct
from data_preproc_functions import create_folder_if_not_exists, get_all_folders, Logger, set_default, sort_human

def get_weeklyct_path(path, weeklyct_folder_name, logger):
    """
    Write dosumentation
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
    Write dosumentation
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
            assert len(patient_weeklyct_path) == 1

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




    

def main():
    save_ct_arr()

if __name__ == '__main__':
    main()