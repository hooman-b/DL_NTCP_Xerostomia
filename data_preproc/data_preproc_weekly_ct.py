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

def save_ct_arr():
    pass

def main():
    save_ct_arr()

if __name__ == '__main__':
    main()