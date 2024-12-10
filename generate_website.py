import json
import pathlib


DIRECTORY = "static/open-scivis-datasets"

TYPE_BYTES = {
    "int16": 2,
    "uint8": 1,
    "uint16": 2,
    "float32": 4,
    "float64": 8,
}


def generate_bibtex(name: str, dataset: dict) -> str:
    if "bibtex" not in dataset or dataset["bibtex"] is None:
        return ""

    bibtex = dataset["bibtex"]
    output = f'@{bibtex["type"]}{{{name},\n'
    for key, val in bibtex.items():
        if key == "type":
            continue
        output += f"    {key} = {{{val}}},\n"
    output += "}"

    return output


def generate_dataset(identifier: str, dataset: dict):
    width, height, depth = dataset["size"]
    box_width, box_height, box_depth = (
        width * dataset["spacing"][0],
        height * dataset["spacing"][1],
        depth * dataset["spacing"][2],
    )

    size = width * height * depth * TYPE_BYTES[dataset["type"]]
    if size > 1024 * 1024 * 1024:
        size = f"{size/1024/1024/1024:.1f} GB"
    elif size > 1024 * 1024:
        size = f"{size/1024/1024:.1f} MB"
    else:
        size = f"{size/1024:.1f} kB"

    # extract preview dimensions from the preview file name
    try:
        preview_file = list(pathlib.Path(f"{DIRECTORY}/{identifier}").glob("preview*"))[0]
        preview_size = preview_file.stat().st_size
        preview_width, preview_height, preview_depth = (
            preview_file.name.removeprefix(f"preview_{identifier}_")
            .removesuffix("_float32.raw")
            .split("x")
        )
    except:
        preview_size = 0
        preview_width, preview_height, preview_depth = 0, 0, 0

    return f"""<details id="{identifier}" data-width="{width}" data-height="{height}" data-depth="{depth}" data-preview-file-size="{preview_size}" data-preview-width="{preview_width}" data-preview-height="{preview_height}" data-preview-depth="{preview_depth}" data-box-width="{box_width}" data-box-height="{box_height}" data-box-depth="{box_depth}">
<summary>
    <span class="name">{dataset['name']}</span>
    <span class="description">{dataset['description']}</span>
    <span class="size">{width}x{height}x{depth} ({size})</span>
    <span class="download"><a href="{dataset['url']}">Download</a></span>
</summary>
<table>
    <tr><th>Description</th><td>{dataset['description']}</td></tr>
    <tr><th>BibTeX</th><td><pre>{generate_bibtex(identifier, dataset)}</pre></td></tr>
    <tr><th>Metadata</th><td><a href="{identifier}/{identifier}.nhdr">NRRD (detached header)</a></td></tr>
    <tr><th>Acknowledgement</th><td>{dataset['acknowledgement']}</td></tr>
    <tr><th>Data type</th><td>{dataset['type']}</td></tr>
    <tr><th>Spacing</th><td>{dataset['spacing'][0]}x{dataset['spacing'][1]}x{dataset['spacing'][2]}</td></tr>
    <tr><th>SHA-512</th><td>{dataset['sha512sum']}</td></tr>
</table>
</details>
"""


def read_dataset(identifier: str) -> dict:
    return json.load(
        open(f"{DIRECTORY}/{identifier}/{identifier}.json", encoding="utf-8")
    )


def generate_page(datasets: dict, categories: list, output_file: str, sort_function):
    identifiers_datasets = list(datasets.items())
    sorted_datasets = sorted(identifiers_datasets, key=sort_function)

    body_html = ""
    for identifier, dataset in sorted_datasets:
        body_html += generate_dataset(identifier, dataset)

    header_html = open("data/header.html", encoding="utf-8").read()
    header_html += "<p><nav>\n"
    for category, identifiers in categories:
        header_html += f'    <a href="category-{category.lower()}.html">{category} ({len(identifiers)})</a>\n'
    header_html += "</nav>\n"
    header_html += '<main id="list">\n'

    footer_html = open("data/footer.html", encoding="utf-8").read()

    with open(f"{DIRECTORY}/{output_file}", "w", encoding="utf-8") as f:
        f.write(header_html + body_html + footer_html)


def generate_nhrd_files(datasets: dict):
    for identifier, dataset in datasets.items():
        # NRRD metadata file (.nhdr)
        with open(
            f"{DIRECTORY}/{identifier}/{identifier}.nhdr", "w", encoding="utf-8"
        ) as f:
            dtype = dataset["type"]
            if dtype == "float32":
                dtype = "float"
            elif dtype == "float64":
                dtype = "double"
            f.write(
                f"""NRRD0004
# Complete NRRD file format specification at:
# http://teem.sourceforge.net/nrrd/format.html
type: {dtype}
dimension: 3
space: left-posterior-superior
sizes: {dataset['size'][0]} {dataset['size'][1]} {dataset['size'][2]}
space directions: ({dataset['spacing'][0]},0,0) (0,{dataset['spacing'][1]},0) (0,0,{dataset['spacing'][2]})
kinds: domain domain domain
endian: little
encoding: raw
space origin: (-{dataset['spacing'][0]*dataset['size'][0]/2},-{dataset['spacing'][1]*dataset['size'][1]/2},-{dataset['spacing'][2]*dataset['size'][2]/2})
data file: {pathlib.Path(dataset['url']).name}

"""
            )  # NRRD format requires a single empty line after the header


def set_urls(url: str, dataset_identifiers: list[str]):
    for identifier in dataset_identifiers:
        dataset = read_dataset(identifier)
        dataset["url"] = (
            f'{url}/{identifier}/{identifier}_{dataset["size"][0]}x{dataset["size"][1]}x{dataset["size"][2]}_{dataset["type"]}.raw'
        )
        json.dump(
            dataset,
            open(f"{DIRECTORY}/{identifier}/{identifier}.json", "w", encoding="utf-8"),
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    config = json.load(open("config.json", encoding="utf-8"))
    url = config["url"]

    # copy data files
    data_files = ["dvr.js", "template.json"]
    for file in data_files:
        src = pathlib.Path(f"data/{file}")
        dst = pathlib.Path(f"{DIRECTORY}/{file}")
        dst.write_bytes(src.read_bytes())

    dataset_identifiers = []
    for path in pathlib.Path(DIRECTORY).iterdir():
        if path.is_dir():
            dataset_identifiers.append(path.name)

    # set url in dataset url field for all json files
    set_urls(url, dataset_identifiers)

    # read datasets (urls may have been updated)
    datasets = {}
    for identifier in dataset_identifiers:
        datasets[identifier] = read_dataset(identifier)

    # generate datasets.json
    json.dump(
        datasets,
        open(f"{DIRECTORY}/datasets.json", "w", encoding="utf-8"),
        indent=4,
        ensure_ascii=False,
    )

    generate_nhrd_files(datasets)

    # discover all categories and count the number of datasets in each
    categories = {}
    for identifier, dataset in datasets.items():
        category = dataset["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(identifier)
    all_categories = [("All", dataset_identifiers)] + sorted(categories.items())

    # sorted alphabetically
    generate_page(datasets, all_categories, "index.html", lambda x: x[1]["name"])

    # sorted by number of voxels
    generate_page(
        datasets,
        all_categories,
        "sorted-by-voxels.html",
        lambda x: x[1]["size"][0] * x[1]["size"][1] * x[1]["size"][2],
    )

    # sorted by size
    generate_page(
        datasets,
        all_categories,
        "sorted-by-size.html",
        lambda x: x[1]["size"][0]
        * x[1]["size"][1]
        * x[1]["size"][2]
        * TYPE_BYTES[x[1]["type"]],
    )

    # generate category pages
    for category, identifiers in all_categories:
        filtered_datasets = {
            identifier: datasets[identifier] for identifier in identifiers
        }
        generate_page(
            filtered_datasets,
            all_categories,
            f"category-{category.lower()}.html",
            lambda x: x[1]["name"],
        )
