import os
import cv2
import numpy as np
from pathlib import Path


def get_project_root() -> Path:
    current_dir = Path(__file__)
    project_dir = [p for p in current_dir.parents if p.parts[-1] == 'wur_navcar'][0]
    return project_dir


def find_file_maxnum(directory):
    files = os.listdir(directory)
    # Find the maximum number in the filenames
    # Filter only files with specific extensions, like jpg, png, npy etc.
    image_files = [file for file in files
                   if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.npy')]
    max_num = max([int(file.split('_')[-1].split('.')[0]) for file in image_files]) if image_files else 0
    return max_num


def set_file_name(directories, key):
    max_num = find_file_maxnum(directories[key])
    if key != 'rgb' and key != 'depth':
        filename = f"{key}_{max_num + 1:02d}"
    else:
        filename = f"{key}_{max_num + 1:02d}.png"

    return filename


def save_joint_cmd(cmd):
    with open('data/cali/j1/cali_j1.txt', 'a') as f:
        np.savetxt(f, np.array([int(cmd)]), newline='\n')


def save_file(directories, file, key):
    filename = set_file_name(directories, key)
    if key == 'rgb':
        cv2.imwrite(os.path.join(directories[key], filename), file)
    elif key == 'depth':
        cv2.imwrite(os.path.join(directories[key], filename), file)
    elif key == 'seg':
        cv2.imwrite(os.path.join(directories[key], filename), file)
    elif key == 'eye_to_hand':
        cv2.imwrite(os.path.join(directories[key], filename), file)
    elif key == 'joint1_nn':
        filename = np.save(os.path.join(directories[key], filename), file)
    elif key == 'imitation_car':
        try:
            existing_data = np.load('data/imitation_car/img_0.npy')
            new_data = np.append(existing_data, file[0], axis=0)
            np.save(os.path.join(directories[key], 'img_0.npy'), new_data)
        except EOFError:
            np.save(os.path.join(directories[key], 'img_0.npy'), file[0])
        with open('data/imitation_car/cmd.txt', 'ab') as f:
            np.savetxt(f, np.array([file[1]]), delimiter=',')
