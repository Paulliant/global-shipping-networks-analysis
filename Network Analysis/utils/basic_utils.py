import yaml
from yaml import Loader
import glob
import os
import pandas as pd
import numpy as np
import cupy as cp
import json
import folium
import random


def get_config(config_path:str):
    """
    Parameters:
        config_path: path to the config file.
    Returns:
        config: config file.
    """
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader)

    return config

def get_all_files(directory_path):

    files_and_directories = os.listdir(directory_path)
    file_paths = []
    for item in files_and_directories:
        item_path = os.path.join(directory_path, item)  
        if os.path.isfile(item_path):
            file_paths.append(item_path)
    return file_paths

def knots_to_kmh(knots):
    """
    将速度从节 (knots) 转换为千米每小时 (km/h)
    :param knots: 速度，单位为节
    :return: 转换后的速度，单位为千米每小时
    """
    kmh = knots * 1.852
    return kmh

def kmh_to_knots(kmh):
    """
    将速度从千米每小时 (km/h) 转换为节 (knots)
    :param kmh: 速度，单位为千米每小时
    :return: 转换后的速度，单位为节
    """
    knots = kmh / 1.852
    return knots

# 定义 Haversine 公式计算距离（千米）
def haversine_distance(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # 地球半径（千米）
    return c * r

# 将 haversine_distance 用 cupy 实现向量化版本
def haversine_distance_gpu(lon1, lat1, lon2, lat2):
    # 将输入转换为 cupy 数组（如果还不是的话）
    lon1, lat1, lon2, lat2 = cp.asarray(lon1), cp.asarray(lat1), cp.asarray(lon2), cp.asarray(lat2)
    lon1, lat1, lon2, lat2 = cp.radians(lon1), cp.radians(lat1), cp.radians(lon2), cp.radians(lat2)
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = cp.sin(dlat/2)**2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon/2)**2
    c = 2 * cp.arcsin(cp.sqrt(a))
    r = 6371  # 地球半径（千米）
    return c * r

# 计算组内距离
def haversine_distance_group(group):
    """
    Parameters:
        group: AIS data grouped by id (MMSI/trips_id).
    Returns:
        distance: distance between two consecutive points in nautical miles.
    
    Typical usage example:

        df = df.sort_values(by=['MMSI', 'time']).reset_index(drop=True)

        df['distance'] = df.groupby('MMSI').apply(haversine_distance).reset_index(drop=True)
    """
    lat1 = group['latitude']
    lon1 = group['longitude']
    lat2 = group['latitude'].shift(1).fillna(0)
    lon2 = group['longitude'].shift(1).fillna(0)
    
    # Convert latitude and longitude to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Approximate radius of the Earth in Nautical miles
    earth_radius = 3440.065

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = earth_radius * c
    distance[0] = 0
    group['distance'] = distance

    return group

def calculate_geodesic_distance_matrix(matrix1:np.ndarray, matrix2:np.ndarray):
    lat1, lon1 = matrix1.T[0], matrix1.T[1]
    lat2, lon2 = matrix2.T[0], matrix2.T[1]
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    # Earth radius in nautical miles
    earth_radius = 3440.065
    
    dlat = lat2[:, np.newaxis] - lat1
    dlon = lon2[:, np.newaxis] - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2[:, np.newaxis]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_matrix = earth_radius * c
    
    return distance_matrix


def find_nearest_port(args):
    """
    根据给定的经纬度坐标，返回最近的港口名称和距离

    params:
        longitude: 经度
        latitude: 纬度
        csv_path: CSV文件路径

    return:
        tuple(最近港口名称, World Port Index Number)
    """
    latitude, longitude, path = args

    try:
        # 读取CSV文件
        df = pd.read_csv(path)

        # 初始化最小距离和最近港口
        min_distance = float('inf')
        nearest_port = -1
        port_code = -1
        water_body=-1

        # 遍历所有港口
        for _, row in df.iterrows():
            # 确保经纬度数据有效
            # 150数据集是Latitude
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                distance = haversine_distance(
                    latitude, longitude,
                    row['Latitude'], row['Longitude']
                )

                # 更新最近的港口
                if distance < min_distance:
                    # country = row['country']
                    # min_distance = distance
                    # nearest_port = row['port']
                    # port_code = row['unlocode']
                    country = row['Country Code']
                    min_distance = distance
                    nearest_port = row['Main Port Name']
                    port_code = row['UN/LOCODE'].replace(" ", "")
                    water_body=row['World Water Body']
                    
        return country, nearest_port, port_code, min_distance, water_body
        # return nearest_port, port_id, min_distance

    # except FileNotFoundError:
    #     return "错误：找不到数据文件", -1,-1,-1,-1
    except Exception as e:
        return f"Error: {str(e)}", -1,-1,-1,-1



def generate_color():
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color



def plot_point(df,columns=['mmsi'],max_width=0):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5,tiles='cartodbpositron')
    mmsis = df['mmsi'].unique()
    colors = {mmsi: generate_color() for mmsi in mmsis}
    for idx, row in df.iterrows():
        mmsi = row['mmsi']
        popup_info = folium.Popup('<br>'.join([f"{col}: {row[col]}" for col in columns if col in df.columns]),max_width=max_width)
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color=colors[mmsi], 
            fill=True,
            popup=popup_info
        ).add_to(m)
    m.add_child(folium.LatLngPopup())
    return m