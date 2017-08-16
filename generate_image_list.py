#!/usr/bin/python
# -*-coding:utf-8-*-

# Generate absolute path list of images for `FlyingThings3D - Scene Flow Dataset`
# Put `generate_image_list.py` under the same directory with `frame_finalpass` folder and `disparity` folder

import os


def generate_image_list(data_dir='frames_finalpass/TRAIN/', label_dir='disparity/TRAIN/'):
    # sub_dirs_data looks like `['C', 'A', 'B']`
    sub_dirs_data = [f1 for f1 in os.listdir(data_dir)
                     if os.path.isdir(os.path.abspath(os.path.join(data_dir, f1)))]
    sub_dirs_labels = [f2 for f2 in os.listdir(label_dir)
                       if os.path.isdir(os.path.abspath(os.path.join(label_dir, f2)))]
    f_train = open('train.lst', 'w')

    assert len(sub_dirs_data) == len(sub_dirs_labels)

    for sub_dir in sub_dirs_data:
        # data_complete_dir looks like `frames_finalpass/TRAIN/A`
        data_complete_dir = os.path.join(data_dir, sub_dir)
        label_complete_dir = os.path.join(label_dir, sub_dir)
        label_index = []
        label_files = []
        data_files = []

        # subdir_num_path looks like `0087`
        for subdir_num_path in os.listdir(label_complete_dir):
            # subdir_num_abs_path looks like `frames_finalpass/TRAIN/C/0087`
	    subdir_num_abs_path = os.path.abspath(os.path.join(label_complete_dir, subdir_num_path))
	    # subdir_left_abs_path looks like `frames_finalpass/TRAIN/C/0087/left`
	    subdir_left_abs_path = str(subdir_num_abs_path) + '/left'

	    # file looks like `0007.pfm`
	    for file in os.listdir(subdir_left_abs_path):
		assert os.path.isfile(str(subdir_num_abs_path) + '/right/' + str(file))
                label_files.append(str(subdir_num_abs_path) + '/left/' + str(file) + '\t' +
				   str(subdir_num_abs_path) + '/right/' + str(file))

        label_files.sort()
        label_files_length = len(label_files)

        # subdir_num_path looks like `0087`
        for subdir_num_path in os.listdir(data_complete_dir):
	    # subdir_num_abs_path looks like `frames_finalpass/TRAIN/C/0087`
	    subdir_num_abs_path = os.path.abspath(os.path.join(data_complete_dir, subdir_num_path))
	    # subdir_left_abs_path looks like `frames_finalpass/TRAIN/C/0087/left`
	    subdir_left_abs_path = str(subdir_num_abs_path) + '/left'

	    # file looks like `0007.png`
	    for file in os.listdir(subdir_left_abs_path):
		assert os.path.isfile(str(subdir_num_abs_path) + '/right/' + str(file))
                data_files.append(str(subdir_num_abs_path) + '/left/' + str(file) + '\t' +
				  str(subdir_num_abs_path) + '/right/' + str(file))

        data_files_length = len(data_files)
	print('data_files_length of folder ' + str(sub_dir) + ': ' + str(data_files_length))
        data_files.sort()

        # The number of labels and data must be the same
        assert label_files_length == data_files_length

        for data_file, label_file in zip(data_files, label_files):
            line = str(data_file) + '\t' + str(label_file) + '\n'
            f_train.write(line)

    f_train.close()
    print("Image list generation completed!")


if __name__ == '__main__':
    generate_image_list()
