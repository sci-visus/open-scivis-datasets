import hashlib
import json
import pathlib
import requests


DIRECTORY = "static/open-scivis-datasets"
MIRROR_URL = "http://klacansky.com/open-scivis-datasets"


def download_raw_file(url, file, sha512sum):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file, "wb") as f:
            m = hashlib.sha512()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                m.update(chunk)
                f.write(chunk)
            assert m.hexdigest() == sha512sum


if __name__ == "__main__":
    for path in pathlib.Path(DIRECTORY).iterdir():
        if path.is_dir():
            identifier = path.name
            dataset = json.load(open(f"{DIRECTORY}/{identifier}/{identifier}.json"))
            raw_file = f'{identifier}_{dataset["size"][0]}x{dataset["size"][1]}x{dataset["size"][2]}_{dataset["type"]}.raw'

            print(identifier, raw_file, flush=True)
            download_raw_file(
                f"{MIRROR_URL}/{identifier}/{raw_file}",
                f"{DIRECTORY}/{identifier}/{raw_file}",
                dataset["sha512sum"],
            )
