import os
import csv
import glob
import pathlib
import numpy as np
import typing as tt
from dataclasses import dataclass
import pandas as pd
from data_process.features import get_feat_split


@dataclass
class Prices:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    
@dataclass
class PriceGold:
    Open: np.ndarray
    Close: np.ndarray
    Returns_Norm: np.ndarray
    Amplitude_Norm: np.ndarray
    Volume_Ratio_Norm: np.ndarray
    Close_Position_Norm: np.ndarray




def csv_to_state_gold(csv_path: pathlib.Path):
    
    df_train, df_val, df_test = get_feat_split(csv_path)
    train_price_gold = PriceGold(
        Open=df_train['Open'].values,
        Close=df_train['Close'].values,
        Returns_Norm=df_train['Returns_Norm'].values,
        Amplitude_Norm=df_train['Amplitude_Norm'].values,
        Volume_Ratio_Norm=df_train['Volume_Ratio_Norm'].values,
        Close_Position_Norm=df_train['Close_Position_Norm'].values,
    )
    val_price_gold = PriceGold(    
        Open=df_val['Open'].values,
        Close=df_val['Close'].values,
        Returns_Norm=df_val['Returns_Norm'].values,
        Amplitude_Norm=df_val['Amplitude_Norm'].values,
        Volume_Ratio_Norm=df_val['Volume_Ratio_Norm'].values,
        Close_Position_Norm=df_val['Close_Position_Norm'].values,
    )
    test_price_gold = PriceGold(    
        Open=df_test['Open'].values,
        Close=df_test['Close'].values,
        Returns_Norm=df_test['Returns_Norm'].values,
        Amplitude_Norm=df_test['Amplitude_Norm'].values,
        Volume_Ratio_Norm=df_test['Volume_Ratio_Norm'].values,
        Close_Position_Norm=df_test['Close_Position_Norm'].values,
    )
    
    
    return train_price_gold, val_price_gold, test_price_gold