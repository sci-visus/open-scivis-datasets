import json
import pathlib

import generate_preview

directory = pathlib.Path("static/open-scivis-datasets")

for dataset_dir in directory.iterdir():
    if not dataset_dir.is_dir():
        continue

    metadata = json.load(
        open(dataset_dir / f"{dataset_dir.name}.json", encoding="utf8")
    )

    filename = pathlib.Path(metadata["url"]).name
    print(filename)
    try:
        generate_preview.generate_preview(dataset_dir / filename, dataset_dir)
    except Exception as e:
        print("error", e)
