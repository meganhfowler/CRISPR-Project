#!/usr/bin/env sh


# Comment out unwanted code

#echo Tuning Hyperparameters...
#./src/tune_hyperparameters.py
#echo Done.
#echo
#echo
echo Training...
./src/train.py
echo Done.
echo
echo
echo Predicting...
./src/predict.py
echo
echo
echo Have a nice day

