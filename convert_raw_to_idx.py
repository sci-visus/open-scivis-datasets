import json
import numpy as np
import pathlib

import OpenVisus as ov

directory = pathlib.Path("static/open-scivis-datasets")

for dataset_dir in directory.iterdir():
    if not dataset_dir.is_dir():
        continue

    metadata = json.load(
        open(dataset_dir / f"{dataset_dir.name}.json", encoding="utf8")
    )

    filename = pathlib.Path(metadata["url"]).name

    data = np.memmap(
        dataset_dir / filename,
        dtype=np.dtype(metadata["type"]),
        mode="r",
        shape=(metadata["size"][2], metadata["size"][1], metadata["size"][0]),
    )
    db = ov.CreateIdx(
        url=str(dataset_dir / f"{dataset_dir.name}.idx"),
        dims=list(reversed(data.shape)),
        fields=[ov.Field("data", metadata["type"])],
        arco="1mb",
    )

    def generate_slices():
        for z in range(data.shape[0]):
            yield data[z, :, :]

    db.writeSlabs(data)
    print(f"Compressing {dataset_dir}")
    db.compressDataset()
