# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys, glob, subprocess, json, math
import random

import numpy as np
import pandas as pd
from scipy.io import wavfile
from os.path import basename, dirname
from tqdm import tqdm
import tempfile, shutil

def id_name(df_metadata, id):
    index = int(id[3:]) - 1
    id_name = str(df_metadata.loc[index]).split('\\t')[-4]
    return id_name

def id_gender(df_metadata, id):
    index = int(id[3:]) - 1
    id_gender = str(df_metadata.loc[index]).split('\\t')[-3]
    return id_gender

def id_nation(df_metadata, id):
    index = int(id[3:]) - 1
    id_nation = str(df_metadata.loc[index]).split('\\t')[-2]
    return id_nation

def rand_crop(audio, len_s):
    start_s = random.randint(0, audio.shape[1] - len_s)
    return audio[:, start_s: start_s+len_s]

def get_filelist1(root_dir):
    fids = []
    for split in ['wav', 'test_wav']:
        all_fns = glob.glob(f"{root_dir}/{split}*/*/*/*.wav")
        for fn in tqdm(all_fns, total=len(all_fns)):
            fids.append('/'.join(fn.split('/')[-4:])[:-4])
    output_fn = f"{root_dir}/file.list"
    with open(output_fn, 'w') as fo:
        fo.write('\n'.join(fids)+'\n')
    return

def get_filelist2(root_dir):
    fids = []
    split = "unzippedIntervalFaces/data"

    all_fns = glob.glob(f"{root_dir}/{split}/*/1.6/*/*/*.jpg")
    for fn in tqdm(all_fns, total=len(all_fns)):
        fids.append('/'.join(fn.split('/')[-6:])[:-5])
    output_fn = f"{root_dir}/file_faces.list"
    with open(output_fn, 'w') as fo:
        fo.write('\n'.join(fids)+'\n')
    return

def get_filelist_withframe(root_dir):
    fids = []
    frames = []
    split = "unzippedIntervalFaces/data"
    df = pd.DataFrame()

    all_fns = glob.glob(f"{root_dir}/{split}/*/1.6/*/*")
    for fn in tqdm(all_fns, total=len(all_fns)):
        fids.append('/'.join(fn.split('/')[-6:]))
        frame = glob.glob(f'{fn}/*.jpg')
        frame_length = len(frame)
        frames.append(frame_length)

    df["path"] = fids
    df["nb_frames"] = frames

    output_fn = f"{root_dir}/file_frame_whole.csv"
    df.to_csv(output_fn)

    return

def prep_faces(root_dir, wav_dir, flist, metadata):
    input_dir, output_dir = root_dir, wav_dir
    os.makedirs(output_dir, exist_ok=True)
    fids = [ln.strip() for ln in open(flist).readlines()]
    vox_metadata = pd.read_csv(metadata).reset_index(drop=False)
    # end_id = len(fids)
    # start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    # fids = fids[0: end_id]
    print(f"{len(fids)} videos")
    for i, fid in enumerate(tqdm(fids)):
        output_mv_dir = "/".join(fid.split("/")[-4:])
        cache_dir = os.path.join(output_dir, output_mv_dir.split("/")[0])
        os.makedirs(cache_dir, exist_ok=True)
        cache_dir = os.path.join(cache_dir, output_mv_dir.split("/")[1])
        os.makedirs(cache_dir, exist_ok=True)
        cache_dir = os.path.join(cache_dir, output_mv_dir.split("/")[2])
        os.makedirs(cache_dir, exist_ok=True)
        cache_dir = os.path.join(cache_dir, output_mv_dir.split("/")[3])
        os.makedirs(cache_dir, exist_ok=True)
        # Get clip info
        clip_name = fid.split('/')[-1].split('.')[0]
        clip_name = str(int(clip_name))
        clip_url = fid.split('/')[-2]
        clip_id = fid.split('/')[-3]
        celeb_name = id_name(vox_metadata, clip_id)
        # celeb_gender = id_gender(vox_metadata, clip_id)
        # celeb_nation = id_nation(vox_metadata, clip_id)
        frame_path = os.path.join(input_dir, "unzippedIntervalFaces/data", celeb_name, '1.6', clip_url, clip_name, "*")
        frame_mv_path = os.path.join(output_dir, output_mv_dir)
        cmd = f"cp -rf {frame_path} {frame_mv_path}"
        subprocess.call(cmd, shell=True)

        # print(f"{video_fn} -> {audio_fn}")
    return


if __name__ == '__main__':
  
    # Get meta: wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
    import argparse
    parser = argparse.ArgumentParser(description='VoxCeleb1 data preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vox', type=str, default="pathto/VoxCeleb/vox1", help='VoxCeleb1 dir')
    parser.add_argument('--metadata', type=str, default='pathto/VoxCeleb/vox1/vox1_meta.csv', help='metadata dir')
    parser.add_argument('--step', type=int, default=2, help='Steps(1: get file list, 2: extract audio)')
    args = parser.parse_args()

    if args.step == 1:
        print(f"Get file list with audio")
        get_filelist1(args.vox)
    elif args.step == 2:
        print(f"Get file list of faces")
        get_filelist_withframe(args.vox)
    elif args.step == 3:
        print(f"Extract audio")
        output_dir = f"{args.vox}/faces"
        manifest = f"{args.vox}/file.list"
        prep_faces(args.vox, output_dir, manifest, args.metadata)
