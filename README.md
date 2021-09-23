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
Set `MODEL` equal to `models.LinearRegressor` or `models.LinearRegressor2`
Set the `MODEL_FILE_PATH` to anything with a `.pkl` extension
Set the `HYPERPARAMS_FILE_PATH` and `PREDICTIONS_FILE_PATH` to anything with a `.csv` extension

By default, the dataframes used are from a previously split crisprSQL dataset. This can be changed by changing the `df` for new training data or the `df_test` for the testing data. The split size of the training data and validation data can also be altered by changing `test_size`.

If you are in the validation stage of the model, set `df_predict = df_validate`. Otherwise if you are in the test stage, set `df_predict = df_test`.
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

### Tune Hyperparameters
Go ro `run.sh` and comment out the rest of the code:
```
echo Tuning Hyperparameters...
./src/tune_hyperparameters.py
echo Done.
echo
echo
#echo Training...
#./src/train.py
#echo Done.
#echo
#echo
#echo Predicting...
#./src/predict.py
#echo
#echo
echo Have a nice day
```
