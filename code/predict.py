#!/usr/bin/env python3
import argparse
import pandas as pd
import models
import torch

# Config
output_file_path = 'test/predictions.csv'

# Load test.csv
df_test = pd.read_csv(data/df_test.csv)

def model_predict(model):
    # Run predictions
    y_predictions = model(model_file_path = 'code/model.pickle').predict(df_test)
    # Save predictions to file
    y_predictions = y_predictions.detach().numpy()
    df_predictions = pd.DataFrame(y_predictions)
    df_predictions.columns = ['prediction']
    df_predictions.to_csv(output_file_path, index=False)

    print(f'{len(y_predictions)} predictions saved to a csv file')

