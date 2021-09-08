#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch

class Preprocessing:
    def select(df, params):
        df = df[params].convert_dtypes()
        return df

    def drop_na(df, params):
        for col in params:
            df = df[df[col].notna()]
        return df

    def remove_dash(df):
        for col in df.select_dtypes(exclude=['number']).columns:
            df[col] = [
                    seq.replace("-", "")
                    for seq in df[col]
                    ]
            return df

    def pad(df):
        for col in df.select_dtypes(exclude=['number']).columns:
            df[col] = df[col].str.pad(width=50, side='right', fillchar='X')
        return df

    def encode_nt(nt: str) -> int:
        assert len(nt) == 1
        encoding_dict = {
                'X':0,
                'A':0.25,
                'T':0.50,
                'G':0.75,
                'C':1.00
                }
        return encoding_dict.get(nt.upper())

    def encode_seq(seq: str):
        encoding = [
                Preprocessing.encode_nt(nt)
                for nt in seq
                ]
        return np.array(encoding)

    def encode(df):
        df["grna_target_sequence"] =  [
                Preprocessing.encode_seq(seq)
                for seq in df["grna_target_sequence"]
                ]
        df["target_sequence"] =  [
                Preprocessing.encode_seq(seq)
                for seq in df["target_sequence"]
                ]
        return df

    def fold_seq(df):
        df["stacked"] = df["grna_target_sequence"].apply(lambda x: x.tolist()) + df["target_sequence"].apply(lambda x: x.tolist())
        df["stacked"] = df["stacked"].apply(lambda x: np.array(x))
        return df["stacked"]

    def tensorfy(stacked):
        temp = []
        for i in stacked:
            temp.append(i)
        return torch.from_numpy(np.array(temp).astype(np.float32))



    def preprocess(df, params):
        df = Preprocessing.select(df, params)
        df = Preprocessing.drop_na(df, params)
        df = Preprocessing.remove_dash(df)
        df = Preprocessing.pad(df)
        df = Preprocessing.encode(df)
        stacked = Preprocessing.fold_seq(df)
        tensor_vec = Preprocessing.tensorfy(stacked)
        return tensor_vec


