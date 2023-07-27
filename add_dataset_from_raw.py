import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import generate_preview


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print(f'Usage: {sys.argv[0]} file.raw name width height depth dtype')
        sys.exit()

    raw_file = sys.argv[1]
    name = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])
    depth = int(sys.argv[5])
    dtype = sys.argv[6]

    config = json.load(open('config.json'))

    # create directory and copy file
    folder = pathlib.Path(f'static/open-scivis-datasets/{name}')
    assert not folder.exists(), f'Folder {folder} already exists, delete it manually before running this script'
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f'{name}_{width}x{height}x{depth}_{dtype}.raw'
    shutil.copy(raw_file, file)

    # compute checksum
    with open(file, 'rb') as f:
        digest = hashlib.file_digest(f, 'sha512')
    sha512sum = digest.hexdigest()

    metadata = {
        'name': name.replace('_', ' ').title(),
        'type': dtype,
        'size': [width, height, depth],
        'spacing': [1, 1, 1],
        'sha512sum': sha512sum,
        'description': '',
        'acknowledgement': '',
        'url': f'{config["url"]}/{name}/{file.name}',
        'date_added': datetime.date.today().isoformat(),
        'category': 'Unknown',
        'bibtex': None,
    }

    with open(folder / f'{name}.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    generate_preview.generate_preview(file, folder)