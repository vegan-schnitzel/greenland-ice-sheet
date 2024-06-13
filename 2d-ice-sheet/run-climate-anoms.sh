#!/usr/bin/env bash

conda activate uib

python 2d_model.py climate-anomalies/deltaT_deltaP/+3
python 2d_model.py climate-anomalies/deltaT_deltaP/+6
python 2d_model.py climate-anomalies/deltaT_deltaP/+9

python 2d_model.py climate-anomalies/deltaT/+3
python 2d_model.py climate-anomalies/deltaT/+6
python 2d_model.py climate-anomalies/deltaT/+9