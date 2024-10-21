import subprocess
import os
from tqdm import tqdm
import ffmpeg
import argparse
from numba import jit,cuda
import numpy as np
from timeit import default_timer as timer


# input_file_path = "/home/mercy/data/compress/"
# output_file_path = "/home/mercy/data/c23/"
# crf_values = [0,23,40]

# for crf_value in crf_values:
#     subprocess.check_output(['ffmpeg', '-i', input_file_path, '-c:v', 'libx264', '-crf', str(crf_value), 
#                     '-preset', 'slow', '-c:a', 'copy', '-y', '-threads', '0', '-nostdin', output_file_path])

# print("Compression complete!")


def compress_video_folder(data_path, output_path, crf, fps):
    codec = 'libx264' if crf != 0 else 'libx264rgb'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i,video in enumerate(tqdm(os.listdir(data_path))):
        if i<2000:
            subprocess.check_output(
                'ffmpeg -r {} -i {} -crf {} -c:v {} -vf "fps={}" {}'.format(
                    str(fps),os.path.join(data_path, video), str(crf), codec,
                    str(fps),
                    os.path.join(output_path, video)),
                shell=True, stderr=subprocess.STDOUT)
    print("Compression complete!")

# compress_video_folder(input_file_path,output_file_path,crf=23,fps=30)
        
def parse_args():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i','--input', type=str, help = 'dataset path')
    parser.add_argument('-o','--output',default=str,help='output path')
    parser.add_argument('-crf','--crf', type=int, default=0)
    parser.add_argument('-fps','--fps', type=int, default=30)
    args = parser.parse_args()      
    return args
  


if __name__ == '__main__':
    args = parse_args()
  
    compress_video_folder(args.input, args.output, args.crf, args.fps)