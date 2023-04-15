#!/bin/bash

pip install diffusers["torch"]
pip install 'transformers[torch]'
python train.py