# CRISPR-Project
Course project for directed research course

## Data Source
Störtz, F. and Minary, P. (2021).  crisprSQL: a novel database platform for crispr/casoff-target cleavage assays._Nucleic Acids Research_,**49**(D1), D855–D861

## Dependencies
* Pandas
* Numpy
* Pytorch
* sklearn
* scipy
* itertools
* pickle
* functools

## Sample Session

### Configuration
Go to `src/config.py`

```
# Select Model and File Path
MODEL = models.LinearRegressor2
MODEL_FILE_PATH = "./src/model2.pkl"
HYPERPARAMS_FILE_PATH = "./results/hyperparameters/model2.csv"
PREDICTIONS_FILE_PATH = "./results/predictions/model2.csv"

# Get dataframes
df = pd.read_csv("./data/train.csv")
df_train, df_validate = train_test_split(df, test_size = 0.2)
df_test = pd.read_csv("./data/test.csv")

# Predict validation or test set?
df_predict = df_validate
```
