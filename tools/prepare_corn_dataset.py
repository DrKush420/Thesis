import os
import shutil
import datetime
import logging
import zipfile
from glob import glob


def create_trainset_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir,'spray'))
    os.makedirs(os.path.join(output_dir,'dont'))
    return output_dir


def extract_from_zip(zip_dir, selection_criteria, period_criteria, unzip_dir, key='.zip'):
    # list zip files in directory
    zip_files = glob(os.path.join(zip_dir, selection_criteria))
    zip_files = [f for f in zip_files if 'labeled_crops' in f]
    start, end = datetime.datetime.strptime(period_criteria.split(':')[0], '%Y-%m-%d'), datetime.datetime.strptime(period_criteria.split(':')[1], '%Y-%m-%d')
    selected_files = []
    for zip_file in zip_files:
        data_date = datetime.datetime.strptime(os.path.basename(zip_file).split('_')[2], '%Y%m%d')
        if data_date >= start and data_date <= end:
            selected_files.append(zip_file)

    print("using the following zip files")
    print([os.path.basename(f) for f in selected_files])
    for zip_file in selected_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)


output_dir = 'data/corn/train_val'
create_trainset_dir(output_dir)
extract_from_zip('data/Corn/2021', '*/labeled*.zip', '2021-05-12:2022-06-30', output_dir)
