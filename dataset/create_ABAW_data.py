import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def take_labels_ex(annot_txt_file):
    with open(annot_txt_file) as f:
        lines = f.readlines()
    labels = []
    for line in lines[1:]:
        labels.append(int(line))
    labels = np.array(labels)
    return labels


def take_labels_au(annot_txt_file):
    with open(annot_txt_file) as f:
        lines = f.readlines()
    labels = []
    for line in lines[1:]:
        labels.append([int(i) for i in line.split(',')])
        # import pdb; pdb.set_trace()
    labels = np.array(labels)
    return labels


def take_labels_va(annot_txt_file):
    with open(annot_txt_file) as f:
        lines = f.readlines()
    labels = []
    for line in lines[1:]:
        labels.append([float(i) for i in line.split(',')])
        # import pdb; pdb.set_trace()
    labels = np.array(labels)
    return labels


def take_images(data_file_name):
    images = glob.glob(data_file_name + '/*.*')
    images.sort()
    number_id = [os.path.split(image)[1].split('.')[0] for image in images]
    number_id = np.array(number_id, int)
    return images, number_id


def creat_df_video(annot_txt_file, data_file_name, path_save, type_partition):
    images_dir, number_id = take_images(data_file_name)
    df = pd.DataFrame(images_dir, columns=['image_id'])

    if type_partition == 'EXPR_Set':
        labels = take_labels_ex(annot_txt_file)
        df['labels_ex'] = labels[number_id - 1]
        df = df[df['labels_ex'].isin(range(9))].reset_index(drop=True)
        df.to_csv(path_save)
    elif type_partition == 'AU_Set':
        labels = take_labels_au(annot_txt_file)
        df['labels_au'] = [list(i) for i in labels[number_id - 1]]
        index_have_negative_value = []
        for index, i in enumerate(df['labels_au']):
            if -1 in i:
                index_have_negative_value.append(index)
        df = df.drop(index=index_have_negative_value).reset_index(drop=True)
        # df['labels_au'].apply(lambda x: str(x))
        df.to_csv(path_save)
    elif type_partition == 'VA_Set':
        labels = take_labels_va(annot_txt_file)
        df['labels_va'] = [list(i) for i in labels[number_id - 1]]
        index_have_negative_value = []
        for index, i in enumerate(df['labels_va']):
            if -5 in i:
                index_have_negative_value.append(index)
        df = df.drop(index=index_have_negative_value).reset_index(drop=True)
        df.to_csv(path_save)


def creat_df_test(path_save, df_test):
    df = pd.read_csv(df_test, index_col=0)
    for i in tqdm(df['folder_dir'], total=len(df)):
        images_dir, number_id = take_images(i)
        df1 = pd.DataFrame(images_dir, columns=['image_id'])
        df1['result'] = 0
        os.makedirs(path_save, exist_ok=True)
        df1.to_csv(path_save + os.path.split(i)[1] + '.csv')


def take_name_video(dir_path):
    return os.path.split(dir_path)[1].split('.')[0]


if __name__ == '__main__':

    dir_images = '/mnt/c/Data/Yuxuan/ABAW/origin_faces/'
    dir_save_df_labels = ['/mnt/c/Data/Yuxuan/ABAW/labels_save/expression/',
                          '/mnt/c/Data/Yuxuan/ABAW/labels_save/action_unit/',
                          '/mnt/c/Data/Yuxuan/ABAW/labels_save/valence_arousal/']

    for set_data in ['Train_Set', 'Validation_Set']:
        for i, type_partition in enumerate(['EXPR_Set', 'AU_Set', 'VA_Set']):
            list_txt = glob.glob(
                f'/mnt/c/Data/Yuxuan/ABAW/annotations/annotations/{type_partition}/{set_data}/*')

            os.makedirs(os.path.join(dir_save_df_labels[i], set_data), exist_ok=True)
            # import pdb; pdb.set_trace()

            for annotation in tqdm(list_txt, total=len(list_txt)):
                path_save = os.path.join(dir_save_df_labels[i], f'{set_data}_v2')
                os.makedirs(path_save, exist_ok=True)
                creat_df_video(annot_txt_file=annotation,
                               data_file_name=os.path.join(dir_images, take_name_video(annotation)),
                               path_save=os.path.join(path_save, take_name_video(annotation) + '.csv'),
                               type_partition=type_partition)

