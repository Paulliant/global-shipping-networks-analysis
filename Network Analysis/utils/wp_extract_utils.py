import time
import pandas as pd
import multiprocessing as mp
from typing import List
from tqdm import tqdm
import utils.basic_utils as basic_utils

def find_stay_point_avg(df,threshold=2):
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
    df['Sog'] = pd.to_numeric(df['Sog'], errors='coerce')
    df['NavigationStatus'] = pd.to_numeric(df['NavigationStatus'], errors='coerce')
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    df['PosTime'] = pd.to_datetime(df['PosTime'], format='mixed', errors='coerce')

    df.dropna(subset=['Lat', 'Lon', 'Sog', 'NavigationStatus', 'ID', 'PosTime'], inplace=True)

    # 对数据按 'ID' 和 'PosTime' 进行排序，保证同一船舶的记录是连续的
    df.sort_values(by=['ID', 'PosTime'], inplace=True)

    df['condition'] = (df['NavigationStatus'] == 5) & (df['Sog'] < threshold)

    ### 点连续平均
    # 创建分组列
    df['group'] = (df['condition'] & (df['condition'] != df['condition'].shift())).cumsum()

    # 过滤符合条件的行
    filtered_df = df[df['condition']]

    # 根据 mmsi 和 group 分组，并计算平均经纬度和平均时间
    result = filtered_df.groupby(['ID', 'group']).agg(
        latitude=('Lat', 'mean'),
        longitude=('Lon', 'mean'),
        timestamp=('PosTime', 'min'),
        timestamp_off=('PosTime', 'max'),
        speed_mean=('Sog','mean'),
        speed_max=('Sog','max')
    ).reset_index()

    # 删除临时的 'group' 列
    result = result.drop(columns=['group'])
    result['timestamp'] = result['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result['timestamp_off'] =  result['timestamp_off'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result['latitude'] /= 600000
    result['longitude'] /= 600000
    
    ####
    ##去除平均值
    # result = df[['latitude','longitude','timestamp','speed']]
    # result['timestamp'] = result['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 为保持输出格式，将 'ID' 列重命名为 'mmsi'
    result = result.rename(columns={'ID': 'mmsi'})
    return result

def process_file(file_path):
    chunksize = 10000000
    results = []
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # 只取 PosAccuracy 和 MsgId 为 1 的记录
            # chunk = chunk[(chunk['PosAccuracy'] == 1) & (chunk['MsgId'].isin([1, 2, 3]))]
            chunk = chunk[(chunk['PosAccuracy'] == 1) & (chunk['MsgId'] == 1)]
            if not chunk.empty:
                res_chunk = find_stay_point_avg(chunk)
                results.append(res_chunk)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['mmsi', 'latitude', 'longitude', 'timestamp', 'timestamp_off', 'speed_mean', 'speed_max'])
    if results:
        final_df = pd.concat(results, ignore_index=True)
        return final_df
    else:
        # 没有符合条件的数据则返回空的 DataFrame（含预期列）
        return pd.DataFrame(columns=['mmsi', 'latitude', 'longitude', 'timestamp', 'timestamp_off', 'speed_mean', 'speed_max'])
    