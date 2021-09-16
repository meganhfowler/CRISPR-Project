#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

import functools

class Preprocessing:
    def drop_na(df, params):
        params.append("cleavage_freq")
        for col in params:
            df = df[df[col].notna()]
        params.remove("cleavage_freq")
        return df

    def select(df, params):
        df = df[params]
        df = df[params].convert_dtypes()
        return df

    def remove_dash(df):
        for col in df.select_dtypes(exclude = ["number"]).columns:
            df[col] = [
                    seq.replace("-", "")
                    for seq in df[col]
                    ]
        return df

    def pad(df):
        for col in df.select_dtypes(exclude = ["number"]).columns:
            df[col] = df[col].str.pad(width = 50, side = "right", fillchar = "X")
        return df

    def encode_nt(nt:str) -> int:
        assert len(nt) == 1
        encoding_dict = {
                'X': 0,
                'A': 0.25,
                'T': 0.5,
                'G': 0.75,
                'C': 1
        }
        return encoding_dict.get(nt.upper())

    def encode_seq(seq:str):
        encoding = [
                Preprocessing.encode_nt(nt)
                for nt in seq
        ]
        return np.array(encoding)

    def encode_col(df, col):
        df[col] = [
            Preprocessing.encode_seq(seq)
            for seq in df[col]
        ]
        return df


    def encode(df):
        for col in df.select_dtypes(exclude = ["number"]).columns:
            Preprocessing.encode_col(df, col)
        return df

    def fold(df):
        df["stacked"] = functools.reduce(lambda x, y: df[x].apply(lambda x: x.tolist()) + df[y].apply(lambda x: x.tolist()),  df.columns)
        return df

    def tensorfy(df):
        temp = []
        for i in df["stacked"]:
            temp.append(i)
        X = torch.from_numpy(np.array(temp).astype(np.float32))
        return X

    def get_X(df, params):
        df = Preprocessing.select(df, params)
        df = Preprocessing.remove_dash(df)
        df = Preprocessing.pad(df)
        df = Preprocessing.encode(df)
        df = Preprocessing.fold(df)
        X = Preprocessing.tensorfy(df)
        return X

    def get_y(df):
        df = df["cleavage_freq"]
        y = torch.Tensor(np.array(df).reshape(df.shape[0], 1).astype(np.float32))
        return y




