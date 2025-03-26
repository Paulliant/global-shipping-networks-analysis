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

name = '01-03'

stay_points_path = f'result/stay_points/stay_points_{name}.csv'
port_path = 'data/UpdatedPub150.csv'
df = pd.read_csv(stay_points_path)

# 使用 apply 方法准备数据
args = df.apply(lambda row: (row['latitude'], row['longitude'], port_path), axis=1)

# 创建多进程池并显示进度条
def process_in_parallel(args):
    with Pool(processes=40) as pool:
        result = list(tqdm(pool.imap(bu.find_nearest_port, args), total=len(args)))
    return result
results = process_in_parallel(args)
df['country'],df['nearest_port'],df['port_code'], df['min_distance'],df['water_body'] = zip(*results)


df.to_csv(f'result/stay_points/stay_points_{name}_updatedpub150.csv',index=False)