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
        parts = line.rstrip('\n').split('~')
        if len(parts) < 5:
            continue
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
            head, line = line.split("~", 1)
            if head == "2":
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
    year = list_source_path[-3]
    month = list_source_path[-2][-2:]
    ais_type = 'STA'
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
    mmsi = int(info[1])
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

def sta_process_line(line, list_mmsi_keep, columns):
    info = line[:-1].split('~')
    if len(info) < 5:
        return None
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

def sta_pos_process(unzip_file_path, list_mmsi_keep, total_lines):
    batch_size = 10000000 # 10000000
    sta_buffer = []
    pos_buffer = []
    sta_columns = ['CommType','ID','ReceiveTime','SourceId','MsgId','IMO','Callsign','ShipName','ShipType','LengthBow','LengthStern','BreadthPort','BreadthStarboard','PosDeviceType','ETA','Draught','Destination','DTE']
    pos_columns = ['CommType','ID','ReceiveTime','SourceID','MsgId','PosTime','Lon','Lat','Cog','Sog','TrueHeading','NavigationStatus','ROT','PosAccuracy','A','B','C','D','E','F','G','H']
    sta_destination_path = '/'.join(unzip_file_path.split('/')[:-1])
    pos_destination_path = sta_destination_path.replace('STA','POS')
    file_name = unzip_file_path.split('/')[-1]
    print(f'    Process file: ' + file_name[:-4])
    sta_save_path = sta_destination_path +  '/' + file_name[:-4] + '.csv'
    pos_save_path = pos_destination_path +  '/' + file_name[:-4] + '.csv'
    
    total_lines = count_lines_with_wc(unzip_file_path)
    
    with open(unzip_file_path,'r') as file:
        df_sta = cudf.DataFrame([], columns=sta_columns)
        df_sta.to_pandas().to_csv(sta_save_path, index=False,header=True)
        df_pos = cudf.DataFrame([], columns=pos_columns)
        df_pos.to_pandas().to_csv(pos_save_path, index=False,header=True)
        for line in tqdm(file,desc="    Processing", total=total_lines, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} | {percentage:3.0f}%"):
        # for line in file:
            head, line = line.split("~", 1)
            if head == "2":
                sta_buffer.append(line)
                if len(sta_buffer) == batch_size: 
                    df_sta = sta_process_buffer(sta_buffer, list_mmsi_keep, sta_columns)
                    df_sta.to_pandas().to_csv(sta_save_path,mode='a',index=False,header=False)
                    sta_buffer = [] # 清空缓冲区
            elif head == "1":
                pos_buffer.append(line)
                if len(pos_buffer) == batch_size: 
                    df_pos = pos_process_buffer(pos_buffer, list_mmsi_keep, pos_columns)
                    df_pos.to_pandas().to_csv(pos_save_path,mode='a',index=False,header=False)
                    pos_buffer = [] # 清空缓冲区

        if sta_buffer:
            df_sta = sta_process_buffer(sta_buffer, list_mmsi_keep, sta_columns)
            df_sta.to_pandas().to_csv(sta_save_path,mode='a',index=False,header=False)
            sta_buffer = []
        if pos_buffer:
            df_pos = pos_process_buffer(pos_buffer, list_mmsi_keep, pos_columns)
            df_pos.to_pandas().to_csv(pos_save_path,mode='a',index=False,header=False)
            pos_buffer = []
    
    return sta_save_path, pos_save_path


if __name__ == "__main__":
    # 传入 源文件位置
    parser = argparse.ArgumentParser()
    parser.add_argument('--year_month', type=str, default="202201")
    args = parser.parse_args()
    directory_path = f'/mnt/nas/fan/ais_ripe_log/{args.year_month[:4]}/{args.year_month[4:6]}/STA'
    # directory_path = '/mnt/nas/fan/ais_ripe_log/2022/202201'
    file_paths = get_all_files(directory_path)
    for source_path in file_paths: # 处理静态文件

        if "ALL_OK_" not in source_path:
            continue

        begin_time = datetime.now()
        print(f'Process: {source_path}. Current Time: {begin_time.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # 获取原始文件名，例如 "ALL_OK_2020-02-01.dat"
        original_filename = os.path.basename(source_path)
        
        # 将原始文件重命名为最后一个"_"后的部分, 例如 "2020-02-01.dat"
        new_filename = original_filename.split('_')[-1]
        
        # 构造新的文件路径，并执行重命名操作
        new_source_path = os.path.join(os.path.dirname(source_path), new_filename)
        os.rename(source_path, new_source_path)

         # 构造目标路径：/mnt/nas/fan/ais_ripe_log/YYYY/MM/STA
        destination_path = f'/mnt/nas/fan/ais_ripe_log/{args.year_month[:4]}/{args.year_month[4:6]}/STA'
        # 从新文件名中去除扩展名，例如 "2020-02-01.dat" 变为 "2020-02-01"
        
        file_name = os.path.splitext(new_filename)[0]
        
        mmsi_keep_path = destination_path + '/' + file_name + '-mmsi-keep.csv'
        unzip_file_path = destination_path + '/' + file_name + '.dat'
        sta_save_path = destination_path +  '/' + file_name + '.csv'
        pos_save_path = destination_path.replace('STA','POS') +  '/' + file_name + '.csv'
        
        if os.path.exists(pos_save_path):
            continue
        if not os.path.exists(sta_save_path):
            if not os.path.exists(unzip_file_path):
                unzip_file_path = copy_and_unzip(source_path=source_path, destination_path=destination_path)
        if not os.path.exists(destination_path.replace('STA','POS')):   # 创建POS的目录
            os.makedirs(destination_path.replace('STA','POS'))
        
        if unzip_file_path != '':
            if not os.path.exists(mmsi_keep_path):
                sta_total_lines, mmsi_keep_path = sta_get_mmsi(unzip_file_path)  #'D:/DATA/Global_AIS/2024/01/STA/2024-01-01-mmsi-keep.csv'
            else:
                sta_total_lines = 1000000
            
            df_mmsi_keep = cudf.read_csv(mmsi_keep_path)
            # list_mmsi_keep = set(df_mmsi_keep['MMSI'].to_list())
            list_mmsi_keep = set(df_mmsi_keep['MMSI'].to_pandas().to_list())
            # 处理静态动态文件
            if not os.path.exists(sta_save_path) or not os.path.exists(pos_save_path):
                sta_save_path, pos_save_path = sta_pos_process(unzip_file_path,list_mmsi_keep, sta_total_lines)
                end_time = datetime.now()
                print(f'    Finish processing {unzip_file_path}.    End Time: {end_time.strftime("%Y-%m-%d %H:%M:%S")}.   Task Duration: {end_time-begin_time}')
                os.remove(unzip_file_path)
            
            end_time = datetime.now()
            print(f'    Finish processing {unzip_file_path}.    End Time: {end_time.strftime("%Y-%m-%d %H:%M:%S")}.   Task Duration: {end_time-begin_time}')