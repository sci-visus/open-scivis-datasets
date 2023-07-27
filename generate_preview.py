import json
import numpy as np
import pathlib
import re
import subprocess
import sys


PRECISION = 8
MAX_VOXELS = 256**3
ZFP_EXE = 'data/zfpcmd.exe'


def generate_preview(raw_file, output_dir):
    p = re.compile(r'(.+)_(\d+)x(\d+)x(\d+)_(.+).raw')
    m = p.match(pathlib.Path(raw_file).name)

    name = m.group(1)
    shape = int(m.group(4)), int(m.group(3)), int(m.group(2))
    dtype = m.group(5)

    # memory mapping used because large datasets may not fit into memory
    data = np.memmap(raw_file, dtype=dtype, mode='r', shape=shape)

    # compute the target low-resolution shape by how many voxels to skip
    stride = 1
    while (shape[0]//stride)*(shape[1]//stride)*(shape[2]//stride) > MAX_VOXELS:
        stride *= 2

    preview = data[::stride, ::stride, ::stride]
    preview.astype(np.float32).tofile('tmp.raw')

    zfp_file = f'{output_dir}/{name}_{preview.shape[2]}x{preview.shape[1]}x{preview.shape[0]}_float{PRECISION}.zfp'

    subprocess.run([ZFP_EXE, '-i', 'tmp.raw', '-f', '-3', str(preview.shape[2]), str(preview.shape[1]), str(preview.shape[0]), '-p',
        str(PRECISION), '-z', zfp_file], check=True)

    pathlib.Path('tmp.raw').unlink()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} input_128x64x32_uint8.raw output_dir')
        sys.exit()

    raw_file = sys.argv[1]
    output_dir = sys.argv[2]

    generate_preview(raw_file, output_dir)
    