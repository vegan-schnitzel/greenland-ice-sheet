#!/usr/bin/env bash

# RUN PDD MODEL SCENARIOS

SCENARIO=$1
PYTHON_ENV=~/miniconda3/envs/uib/bin/python
EXP_FOLDER=~/MSc/Erasmus/curriculum/glaciology/assignments

if [[ $# -eq 0 ]] ; then
    echo 'choose scenario via command-line'
    exit 0
fi

# copy simulation, otherwise new parameter list is ignored (?)
cp ${EXP_FOLDER}/pdd-model/pdd-model.py \
   ${EXP_FOLDER}/climate-change-scenarios/${SCENARIO}/
# link model-input
ln -sf ${EXP_FOLDER}/model-input \
       ${EXP_FOLDER}/climate-change-scenarios/

# jump into simulation scenario
cd ${EXP_FOLDER}/climate-change-scenarios/${SCENARIO}
# run pdd simulation using parameters defined in
# scenario folder!
echo 'simulating' $SCENARIO
$PYTHON_ENV pdd-model.py | tee output.log