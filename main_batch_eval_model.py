from os import listdir, system, remove
from os.path import isfile, join, basename, exists
import time
import re
import pandas as pd

MODEL_IDX = 0
POSITION_MEAN_IDX = 1
POSITION_MEDIAN_IDX = 3
ORIRENT_MEAN_IDX = 2
ORIRENT_MEDIAN_IDX = 4

import numpy as np

input_path = '/media/dev/storage1/robotcar_lr1e6/RobotCar_full_HyperAtLoc_False'
data_dir = '/media/dev/storage1'
path = join(input_path, 'models')
scene = 'full'
scene_idx = 2
model_type = 'HyperAtLoc'
gpu_to_run = 1
test_model_freq = 1
min_epoch_to_eval = 0
models = sorted([join(path, f) for f in listdir(path) if isfile(join(path, f))])

pose_error = np.zeros((len(models), 5))
model_eval_count = 0

output_file = f'res_{model_type}_{time.strftime("%y_%m_%d_%H_%M_%S", time.localtime())}.txt'
if exists(output_file):
    remove(output_file)

for model in models:
    m = re.match('epoch_([\d]+).pth.tar', basename(model))
    model_idx = int(m.group(1)) if m is not None else -1
    if model_idx < min_epoch_to_eval or (model_idx - min_epoch_to_eval) % test_model_freq:
        print(f'skipping file: {basename(model)}')
        continue

    print(f'Running model evaluation: {basename(model)}')
    print('\t- Model evaluation script invoked')
    eval_cmd_str = f'python eval.py --dataset RobotCar --scene {scene} --model {model_type} --gpus {gpu_to_run} --weights {model} --data_dir {data_dir} > {output_file}'
    system(eval_cmd_str)
    print('\t- Model evaluation script is done')

    with open(output_file) as fp:
        Lines = fp.readlines()
        pose_error[model_eval_count, MODEL_IDX] = int(model_idx)

        for line in Lines:
            m = re.match('Error in translation: median ([\d.]+) m,  mean ([\d.]+) m ', line)
            if m is not None:
                pose_error[model_eval_count, POSITION_MEDIAN_IDX] = float(m.group(1))
                pose_error[model_eval_count, POSITION_MEAN_IDX] = float(m.group(2))

            m = re.match('Error in rotation: median ([\d.]+) degrees, mean ([\d.]+) degree', line)
            if m is not None:
                pose_error[model_eval_count, ORIRENT_MEDIAN_IDX] = float(m.group(1))
                pose_error[model_eval_count, ORIRENT_MEAN_IDX] = float(m.group(2))

            if 'Running sequences:' in line:
                eval_sqe_line = line

    print(f'\t- {eval_sqe_line}')
    print(f'\t- Results retrieved: {pose_error[model_eval_count, :]}')
    model_eval_count += 1

    if exists(output_file):
        remove(output_file)

    batch_eval_res = pd.DataFrame(pose_error[:model_eval_count, :],
                                  columns=['epoch', 'trans_mean', 'orient_mean', 'trans_median', 'orient_median'])
    output_csv_filename = f'{model_type}_robotCar_{scene}_{scene_idx}_batch_eval.csv'
    batch_eval_res.to_csv(join(input_path, output_csv_filename), index=False)
    print(f'\t- Saved updated evaluation results: {output_csv_filename}')
