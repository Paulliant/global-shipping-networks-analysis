# import pandas as cudf
import cudf
import os
from tqdm import tqdm
import re
import shutil
import tarfile
import os
import concurrent.futures
from datetime import datetime
import subprocess
import warnings
import argparse
warnings.filterwarnings("ignore")

def sta_mmsi_process_lines(buffer):
    ship_info_5 = []
    ship_info_19 = []
    ship_info_24 = []
    for line in buffer:
        msgid = line.split('~')[4]
        info_list = line[:-1].split('~')
        # 只保留mmsi,msgid, ShipType
        if msgid == '5':
            info = [info_list[i] for i in [1, 4, 8, 7, 6]]
            ship_info_5.append(info)
        elif msgid == '19':
            info = [info_list[i] for i in [1, 4, 6, 5]]
            info.append('')
            ship_info_19.append(info)
        elif msgid == '24B':
            info = [info_list[i] for i in [1, 4, 5]]
            info.append('')
            info.append(info_list[7])
            ship_info_24.append(info)
        else:
            continue
    return ship_info_5, ship_info_19, ship_info_24

def count_lines_with_wc(file_path): 
    # 调用 wc -l 指令 
    result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True) 
    # 解析输出并获取行数 
    line_count = int(result.stdout.split()[0]) 
    return line_count

def sta_mmsi_process_buffer(buffer):
    ship_info_5, ship_info_19, ship_info_24 = sta_mmsi_process_lines(buffer)
    ship_info = ship_info_5
    ship_info.extend(ship_info_19)
    ship_info.extend(ship_info_24)
    
    df_ship_info = cudf.DataFrame(ship_info, columns=['MMSI', 'MsgId', 'ShipType', 'ShipName', 'Callsign'])
    df_ship_info.drop_duplicates(inplace=True)
    
    pattern = re.compile(r'^[2-7]')
    df = df_ship_info
    df = df[(df['ShipType'] != '')]
    df = df[df['MMSI'].astype(str).str.match(pattern) & (df['MMSI'].astype(str).str.len() == 9)].reset_index(drop=True)
    count = df.groupby('MMSI')['ShipType'].nunique() 
    duplicated_values = count[count > 1].index
    df_result = df[~df['MMSI'].isin(duplicated_values)]
    
    # df_result.to_csv(save_path,mode='a',index=False,header=True)
    return df_result

def copy_and_unzip(source_path, destination_path): 
    # 拷贝文件从路径A到路径B 
    if not os.path.exists(source_path): 
        print(f"\033File {source_path} 不存在") 
        return ''
    if not os.path.exists(destination_path): 
        os.makedirs(destination_path)
    shutil.copy(source_path, destination_path) 
    print(f"    Copy {source_path} To {destination_path}.") 
    # 解压缩文件到指定目录 
    file_name = source_path.split('/')[-1]
    with tarfile.open(destination_path + '/' + file_name, 'r:gz') as tar_ref: 
        tar_ref.extractall(destination_path)
    print(f"    Unzip {destination_path}/{file_name}.")
    os.remove(destination_path + '/' + file_name)
    unzip_path = destination_path + '/' + file_name.split('.')[0] + '.dat'
    new_file_path = destination_path + '/' + file_name.split('.')[0].split('_')[-1] + '.dat'
    os.rename(unzip_path, new_file_path)
    return new_file_path

def sta_get_mmsi(unzip_file_path):
    batch_size = 10000000
    buffer = []
    destination_path = '/'.join(unzip_file_path.split('/')[:-1])
    file_name = unzip_file_path.split('/')[-1]
    print(f'    Process sta file mmsi: ' + file_name[:-4])
    save_path = destination_path +  '/' + file_name[:-4] + '-mmsi.csv'
    
    # with open(unzip_file_path, 'r') as file: # 计算文件总行数 
    #     sta_total_lines = sum(1 for line in file)
    sta_total_lines = count_lines_with_wc(unzip_file_path)
    
    with open(unzip_file_path,'r') as file:
        df_ship_info = cudf.DataFrame([], columns=['MMSI', 'MsgId', 'ShipType', 'ShipName', 'Callsign'])
        df_ship_info.to_pandas().to_csv(save_path, index=False,header=True)
        for line in tqdm(file,desc="    Processing", total=sta_total_lines, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} | {percentage:3.0f}%"):
        # for line in file:
            buffer.append(line) 
            if len(buffer) == batch_size: 
                df_ship_info = sta_mmsi_process_buffer(buffer)
                df_ship_info.to_pandas().to_csv(save_path,mode='a',index=False,header=False)
                buffer = [] # 清空缓冲区

        if buffer:
            df_ship_info = sta_mmsi_process_buffer(buffer)
            df_ship_info.to_pandas().to_csv(save_path,mode='a',index=False,header=False)
            buffer = [] 
    
    # 存储60-89的mmsi
    mmsi_keep_path = destination_path +  '/' + file_name[:-4] + '-mmsi-keep.csv'
    df = cudf.read_csv(save_path)
    df['ShipType'] = cudf.to_numeric(df['ShipType'], errors='coerce')
    original_length = len(df)
    df = df[(df['ShipType']< 90) & (df['ShipType']> 59)]
    filter_length = len(df)
    df.to_csv(mmsi_keep_path,index=False, header=True)
    print(f'    Length after filtered: {filter_length} / {original_length}, {(filter_length/original_length):.2f}')
    return sta_total_lines, mmsi_keep_path

def parse_source_path(source_path):
    """
    提取路径中的文件名，生成用于存储处理结果的目标路径
    """
    list_source_path = source_path.split('/')
    file_name = list_source_path[-1].split('.')[0].split('_')[-1]
    year = list_source_path[-4]
    month = list_source_path[-3][-2:]
    ais_type = list_source_path[-2]
    destination_path = '/mnt/nas/fan/ais_ripe_log/' + year + '/' + month + '/' + ais_type    # D:\DATA\Global_AIS\2024\01\STA
    return file_name, destination_path

def get_all_files(directory_path):
    files_and_directories = os.listdir(directory_path)
    file_paths = []
    for item in files_and_directories:
        item_path = os.path.join(directory_path, item)  
        if os.path.isfile(item_path):
            file_paths.append(item_path)
    return file_paths



def pos_process_line(line, list_mmsi_keep, columns):
    info = line[:-1].split('~')
    if len(info) < len(columns): 
        info.extend([''] * (len(columns) - len(info)))
    elif len(info) > len(columns):
        info = info[:len(columns)]
    dtype = info[0]
    try:
        mmsi = int(info[1])
    except ValueError:
        # 非法 mmsi，直接忽略这一行
        return None
    if dtype == '1' and (mmsi in list_mmsi_keep):
        return info

def pos_process_buffer(buffer, list_mmsi_keep, columns):
    commen_type_1 = [] 
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor: 
        results = list(executor.map(pos_process_line, buffer, [list_mmsi_keep]*len(buffer), [columns]*len(buffer))) 
        commen_type_1 = [result for result in results if result is not None]
            
    df = cudf.DataFrame(commen_type_1,columns=columns)
    df['ReceiveTime'] = cudf.to_numeric(df['ReceiveTime'], errors='coerce')
    df['ReceiveTime'] = cudf.to_datetime(df['ReceiveTime'],unit='s')
    df['PosTime'] = cudf.to_numeric(df['PosTime'], errors='coerce') # 将字符串转换为数值类型 
    df['PosTime'] = cudf.to_datetime(df['PosTime'], unit='s') # 转换为日期时间
    return df
  
def pos_process(unzip_file_path, list_mmsi_keep):
    batch_size = 10000000 # 10000000
    columns = ['CommType','ID','ReceiveTime','SourceID','MsgId','PosTime','Lon','Lat','Cog','Sog','TrueHeading','NavigationStatus','ROT','PosAccuracy','A','B','C','D','E','F','G','H']
    buffer = []
    destination_path = '/'.join(unzip_file_path.split('/')[:-1])
    file_name = unzip_file_path.split('/')[-1]
    print(f'    Process pos file: ' + file_name[:-4])
    save_path = destination_path +  '/' + file_name[:-4] + '.csv'
    
    # with open(unzip_file_path, 'r') as file: # 计算文件总行数 
    #     total_lines = sum(1 for line in file)
    total_lines = count_lines_with_wc(unzip_file_path)
    
    with open(unzip_file_path,'r') as file:
        df_pos = cudf.DataFrame([], columns=columns)
        df_pos.to_csv(save_path, index=False,header=True)
        for line in tqdm(file,desc="    Processing", total=total_lines, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} | {percentage:3.0f}%"):
        # for line in file:
            buffer.append(line) 
            # print(line)
            if len(buffer) == batch_size: 
                df_pos = pos_process_buffer(buffer, list_mmsi_keep, columns)
                df_pos.to_pandas().to_csv(save_path,mode='a',index=False,header=False)
                buffer = [] # 清空缓冲区

        if buffer:
            df_pos = pos_process_buffer(buffer, list_mmsi_keep, columns)
            df_pos.to_pandas().to_csv(save_path,mode='a',index=False,header=False)
            buffer = []
    
    return save_path


def sta_process_line(line, list_mmsi_keep, columns):
    info = line[:-1].split('~')
    msgid = info[4]
    if len(info) < len(columns): 
        info.extend([''] * (len(columns) - len(info)))
    mmsi = int(info[1])
    if msgid == '5' and (mmsi in list_mmsi_keep):
        return info


def sta_process_buffer(buffer, list_mmsi_keep, columns):
    msgid_5 = [] 
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor: 
        results = list(executor.map(sta_process_line, buffer, [list_mmsi_keep]*len(buffer), [columns]*len(buffer))) 
        msgid_5 = [result for result in results if result is not None]
            
    df = cudf.DataFrame(msgid_5,columns=columns)
    df['ReceiveTime'] = cudf.to_numeric(df['ReceiveTime'], errors='coerce')
    df['ReceiveTime'] = cudf.to_datetime(df['ReceiveTime'],unit='s')
    return df

def sta_process(unzip_file_path, list_mmsi_keep, sta_total_lines):
    batch_size = 10000000 # 10000000
    buffer = []
    columns = ['CommType','ID','ReceiveTime','SourceId','MsgId','IMO','Callsign','ShipName','ShipType','LengthBow','LengthStern','BreadthPort','BreadthStarboard','PosDeviceType','ETA','Draught','Destination','DTE']
    destination_path = '/'.join(unzip_file_path.split('/')[:-1])
    file_name = unzip_file_path.split('/')[-1]
    print(f'    Process sta file: ' + file_name[:-4])
    save_path = destination_path +  '/' + file_name[:-4] + '.csv'
    
    sta_total_lines = count_lines_with_wc(unzip_file_path)
    
    with open(unzip_file_path,'r') as file:
        df_sta = cudf.DataFrame([], columns=columns)
        df_sta.to_pandas().to_csv(save_path, index=False,header=True)
        for line in tqdm(file,desc="    Processing", total=sta_total_lines, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} | {percentage:3.0f}%"):
        # for line in file:
            buffer.append(line) 
            if len(buffer) == batch_size: 
                df_sta = sta_process_buffer(buffer, list_mmsi_keep, columns)
                df_sta.to_pandas().to_csv(save_path,mode='a',index=False,header=False)
                buffer = [] # 清空缓冲区

        if buffer:
            df_sta = sta_process_buffer(buffer, list_mmsi_keep, columns)
            df_sta.to_pandas().to_csv(save_path,mode='a',index=False,header=False)
            buffer = [] 
    
    return save_path


if __name__ == "__main__":
    # 传入 源文件位置
    parser = argparse.ArgumentParser()
    parser.add_argument('--year_month', type=str, default="202301")
    args = parser.parse_args()
    directory_pos_path = f'/mnt/nas/fan/ais_ripe_log/{args.year_month[:4]}/{args.year_month}/POS'
    # directory_pos_path = '/mnt/e/ais_ripe_log/2024/202401/POS'
    directory_sta_path = directory_pos_path[:-3] + 'STA'
    file_sta_paths = get_all_files(directory_sta_path)
    for source_path in file_sta_paths: # 处理静态文件压缩包
        begin_time = datetime.now()
        print(f'Process: {source_path}. Current Time: {begin_time.strftime("%Y-%m-%d %H:%M:%S")}')
        file_name, destination_path= parse_source_path(source_path)
        
        mmsi_keep_path = destination_path + '/' + file_name + '-mmsi-keep.csv'
        unzip_sta_file_path = destination_path + '/' + file_name + '.dat'
        sta_save_path = destination_path +  '/' + file_name + '.csv'
        unzip_pos_file_path = destination_path.replace('STA','POS') + '/'  + file_name + '.dat'
        pos_save_path = destination_path.replace('STA','POS') +  '/' + file_name + '.csv'
        
        if os.path.exists(pos_save_path):
            continue
        if not os.path.exists(sta_save_path):
            if not os.path.exists(unzip_sta_file_path):
                unzip_sta_file_path = copy_and_unzip(source_path=source_path, destination_path=destination_path)
        
        if unzip_sta_file_path != '':
            if not os.path.exists(mmsi_keep_path):
                sta_total_lines, mmsi_keep_path = sta_get_mmsi(unzip_sta_file_path)  #'D:/DATA/Global_AIS/2024/01/STA/2024-01-01-mmsi-keep.csv'
            else:
                sta_total_lines = 1000000
            
            df_mmsi_keep = cudf.read_csv(mmsi_keep_path)
            # list_mmsi_keep = set(df_mmsi_keep['MMSI'].to_list())
            list_mmsi_keep = set(df_mmsi_keep['MMSI'].to_pandas().to_list())
            # 处理静态文件
            if not os.path.exists(sta_save_path):
                sta_save_path = sta_process(unzip_sta_file_path,list_mmsi_keep, sta_total_lines)
                end_time = datetime.now()
                print(f'    Finish processing {unzip_sta_file_path}.    End Time: {end_time.strftime("%Y-%m-%d %H:%M:%S")}.   Task Duration: {end_time-begin_time}')
                os.remove(unzip_sta_file_path)
            
            source_pos_path = source_path.replace('STA','POS')
            destination_path= destination_path.replace('STA','POS')    #'D:/DATA/Global_AIS/2024/01/STA'
            if not os.path.exists(pos_save_path):
                if not os.path.exists(unzip_pos_file_path):
                    unzip_pos_file_path = copy_and_unzip(source_path=source_pos_path, destination_path=destination_path)    #'D:/DATA/Global_AIS/2024/01/POS/2024-01-01.dat'
            if unzip_pos_file_path!= '':
                # 读取动态文件
                if not os.path.exists(pos_save_path):
                    pos_save_path = pos_process(unzip_pos_file_path, list_mmsi_keep)
                    os.remove(unzip_pos_file_path)
                    
            end_time = datetime.now()
            print(f'    Finish processing {unzip_pos_file_path}.    End Time: {end_time.strftime("%Y-%m-%d %H:%M:%S")}.   Task Duration: {end_time-begin_time}')