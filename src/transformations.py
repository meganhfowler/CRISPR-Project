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
        df[params] = df[params].convert_dtypes()
        return df

    def symbol_mapping(sym):
        mapping = {'+': 1.00, '-': 0.00}
        return mapping.get(sym)

    def encode_strand(df):
        df["target_strand"] = [
            Preprocessing.symbol_mapping(sym)
            for sym in df["target_strand"]
        ]
        df["grna_target_strand"] = [
            Preprocessing.symbol_mapping(sym)
            for sym in df["grna_target_strand"]
        ]
        return df

    def study_mapping(name):
        mapping = {
            'Tsai_circle': 0.00/16.00,
            'Finkelstein': 1.00/16.00,
            'Tsai': 2.00/16.00,
            'Cameron': 3.00/16.00,
            'Kleinstiver': 4.00/16.00,
            'Slaymaker': 5.00/16.00,
            'Kim16': 6.00/16.00,
            'Ran': 7.00/16.00,
            'Anderson': 8.00/16.00,
            'KimChromatin': 9.00/16.00,
            'Chen17': 10.00/16.00,
            'Listgarten': 11.00/16.00,
            'Cho': 12.00/16.00,
            'Kim': 13.00/16.00,
            'Fu': 14.00/16.00,
            'Frock': 15.00/16.00,
            'Wang': 16.00/16.00
        }
        return mapping.get(name)

    def encode_study(df):
        df["study_name"] = [
            Preprocessing.study_mapping(name)
            for name in df["study_name"]
        ]
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

    def encode_nt_onehot(nt:str) -> int:
        assert len(nt) == 1
        encoding_dict = {
            'X': [0, 0, 0, 0],
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'C': [0, 0, 0, 1]
        }
        return encoding_dict.get(nt.upper())

    def encode_seq_onehot(seq:str):
        encoding = [
            Preprocessing.encode_nt_onehot(nt)
            for nt in seq
        ]
        encoding = np.asarray(encoding).flatten()
        return np.array(encoding)

    def encode_col_onehot(df, col):
        encoded = [
            Preprocessing.encode_seq_onehot(seq)
            for seq in df[col]
        ]
        return encoded

    def fold(df):
        df["stacked"] = functools.reduce(lambda x, y: df[x].apply(lambda x: x.tolist()) + df[y].apply(lambda x: x.tolist()),  df.columns)
        return df

    def fold_2(df):
        target_seq = Preprocessing.encode_col_onehot(df, "target_sequence")
        grna_target_seq = Preprocessing.encode_col_onehot(df, "grna_target_sequence")

        target_seq = np.asarray(target_seq, dtype = np.float32)
        grna_target_seq = np.asarray(grna_target_seq, dtype = np.float32)
        seqs = np.concatenate((target_seq, grna_target_seq), axis = 1)
        target_strand = np.asarray(df["target_strand"], dtype = np.float32)
        grna_target_strand = np.asarray(df["grna_target_strand"], dtype = np.float32)
        strands = zip(target_strand, grna_target_strand)
        strands = tuple(strands)
        strands = np.asarray(strands, dtype = np.float32)
        e1 = df["energy_1"]
        e2 = df["energy_2"]
        e3 = df["energy_3"]
        e4 = df["energy_4"]
        e5 = df["energy_5"]

        energies = zip(e1, e2, e3, e4, e5)
        energies = tuple(energies)
        energies = np.asarray(energies, dtype = np.float32)

        study_name = df["study_name"]
        delivery_mode = df["delivery_mode"]
        whole_genome = df["whole_genome"]

        study_details = zip(study_name, delivery_mode, whole_genome)
        study_details = tuple(study_details)
        study_details = np.asarray(study_details, dtype = np.float32)

        all_data = np.concatenate((strands, seqs, energies, study_details), axis = 1)

        X = torch.from_numpy(all_data)
        return X



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


    def get_X_2(df, params):
        df = Preprocessing.select(df, params)
        df = Preprocessing.encode_strand(df)
        df = Preprocessing.encode_study(df)
        df = Preprocessing.remove_dash(df)
        df = Preprocessing.pad(df)
        X = Preprocessing.fold_2(df)
        return X



