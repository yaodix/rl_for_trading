#!/usr/bin/env python3
"""
生成30分钟K线数据（纯Python实现）
输入：1分钟K线CSV文件，格式为：
    datetime_str,timestamp,open,high,low,close,volume,amount
输出：30分钟K线CSV文件，格式为：
    symbol,trade_time,open,high,low,close,volume,amount
"""

import csv
from datetime import datetime, time, timedelta
from collections import defaultdict
import sys
import pandas as pd

# 交易时段30分钟区间定义 (start, end)，左闭右开 (start <= dt.time < end)
INTERVALS = [
    (time(9, 30), time(10, 0)),   # 结束 10:00
    (time(10, 0), time(10, 30)),  # 结束 10:30
    (time(10, 30), time(11, 0)),  # 结束 11:00
    (time(11, 0), time(11, 30)),  # 结束 11:30
    (time(13, 0), time(13, 30)),  # 结束 13:30
    (time(13, 30), time(14, 0)),  # 结束 14:00
    (time(14, 0), time(14, 30)),  # 结束 14:30
    (time(14, 30), time(15, 0)),  # 结束 15:00
]

def is_trading_time(t: time) -> bool:
    """判断时间是否在交易时段内"""
    # 上午 9:30 - 11:30
    if time(9, 30) <= t <= time(11, 30):
        return True
    # 下午 13:00 - 15:00
    if time(13, 0) <= t <= time(15, 0):
        return True
    return False

def get_interval_end(dt: datetime) -> datetime:
    """返回该datetime所属的30分钟区间结束时间"""
    t = dt.time()
    # 检查每个区间
    for start, end in INTERVALS:
        if start <= t < end:
            # 组合日期和结束时间
            return datetime.combine(dt.date(), end)
    # 如果没有匹配（理论上不应该发生，因为已经过滤了交易时段）
    raise ValueError(f"时间 {dt} 不在任何30分钟区间内")

def process_file(input_path: str, output_path: str):
    """处理主函数"""
    # 用于存储分组数据：键为区间结束时间，值为该区间内的记录列表
    buckets = defaultdict(list)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过标题行
        # 检查列格式
        if header[0] == '':
            # 第一列为空，调整列索引
            col_idx = {'datetime': 0, 'timestamp': 1, 'open': 2, 'high': 3, 'low': 4, 'close': 5, 'volume': 6, 'amount': 7}
        else:
            # 假设列顺序相同
            col_idx = {'datetime': 0, 'timestamp': 1, 'open': 2, 'high': 3, 'low': 4, 'close': 5, 'volume': 6, 'amount': 7}
        
        for row in reader:
            if len(row) < 8:
                continue
            dt_str = row[col_idx['datetime']]
            # 解析日期时间
            try:
                dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S')
            except ValueError:
                print(f"警告：无法解析日期时间 {dt_str}，跳过")
                continue
            
            # 过滤非交易时段
            if not is_trading_time(dt.time()):
                continue
            
            # 获取区间结束时间
            try:
                bin_end = get_interval_end(dt)
            except ValueError as e:
                # 忽略不在区间内的时间（例如 9:30之前）
                continue
            
            # 读取数值
            try:
                open_price = float(row[col_idx['open']])
                high = float(row[col_idx['high']])
                low = float(row[col_idx['low']])
                close = float(row[col_idx['close']])
                volume = float(row[col_idx['volume']])
                amount = float(row[col_idx['amount']])
            except ValueError:
                print(f"警告：数值转换失败 {row}，跳过")
                continue
            
            # 存储到对应的桶
            buckets[bin_end].append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'amount': amount,
                'datetime': dt  # 用于排序
            })
    
    if not buckets:
        print("错误：没有可聚合的数据")
        return
    
    # 对每个桶进行聚合
    results = []
    for bin_end, records in buckets.items():
        # 按时间排序（确保第一分钟和最后一分钟正确）
        records.sort(key=lambda x: x['datetime'])
        open_price = records[0]['open']
        close_price = records[-1]['close']
        high = max(r['high'] for r in records)
        low = min(r['low'] for r in records)
        volume = sum(r['volume'] for r in records)
        amount = sum(r['amount'] for r in records)
        
        results.append({
            'symbol': '518880.SH',
            'trade_time': bin_end.strftime('%Y-%m-%d %H:%M:%S'),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume,
            'amount': amount,
        })
    
    # 按交易时间排序
    results.sort(key=lambda x: x['trade_time'])
    
    # 写入CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['symbol', 'trade_time', 'open', 'high', 'low', 'close', 'volume', 'amount']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"已生成 {len(results)} 条30分钟K线数据，保存至 {output_path}")

if __name__ == '__main__':
    # 默认输入输出路径
    input_file = 'Rl_trading_demo/project_one/data/518880.SH.1m.csv'
    output_file = 'Rl_trading_demo/project_one/data/518880.SH.30m.csv'
    
    # 允许命令行参数覆盖
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    process_file(input_file, output_file)