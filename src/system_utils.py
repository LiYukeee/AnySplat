import numpy as np
import subprocess
import os

def autoChooseCudaDevice():
    # set default CUDA device according to GPU memory
    try:
        cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
        os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
        print(f'--- Auto chose CUDA device {os.environ["CUDA_VISIBLE_DEVICES"]} ---')
    except:
        print(f'--- Auto chose CUDA device Error ---')
