import pandas as pd
import os
from tqdm import tqdm
import time
import numpy as np
import folium
import utils.basic_utils as bu
from multiprocessing import Pool
from utils.wp_extract_utils import process_file

import warnings
warnings.filterwarnings("ignore")

import datetime

name = '01-03__'

data_dictionary_01 = '/mnt/nas/fan/ais_ripe_log/2022/01/POS'
data_dictionary_02 = '/mnt/nas/fan/ais_ripe_log/2022/02/POS'
data_dictionary_03 = '/mnt/nas/fan/ais_ripe_log/2022/03/POS'
# port_path = 'data/ports_info.csv'
port_path = 'data/UpdatedPub150.csv'

def process_in_parallel(file_paths):
    with Pool(processes=5) as pool:
        # result = list(pool.imap(process_file, file_paths))
        result = list(tqdm(pool.imap(process_file, file_paths), total=len(file_paths)))
    results = pd.concat(result, ignore_index=True)
    results = results.reset_index(drop=True)
    return results

file_paths = bu.get_all_files(data_dictionary_01) + bu.get_all_files(data_dictionary_02) + bu.get_all_files(data_dictionary_03)
# file_paths = file_paths[:1]

print("Process start time:", datetime.datetime.now())
final_result = process_in_parallel(file_paths)
final_result.sort_values(by=['mmsi', 'timestamp'], inplace=True)

print("Process end time:", datetime.datetime.now())

final_result.to_csv(f'result/stay_points/stay_points_{name}.csv',index=False)