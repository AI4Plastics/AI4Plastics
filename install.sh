#!/bin/bash

ENV_PATH=./.env

conda env create --file environment_A100.yml -p $ENV_PATH
source activate $ENV_PATH
