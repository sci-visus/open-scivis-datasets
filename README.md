All scripts and metadata necessary to deploy the [Open Scivis Datasets](https://klacansky.com/open-scivis-datasets) website. Scripts to create
[OpenVisus](https://github.com/sci-visus/OpenVisus) IDX files are missing because this feature is experimental.

The scripts were tested on Python 3.13 and Windows 11. 

## Datasets Mirrors

[Open Scivis Datasets](https://klacansky.com/open-scivis-datasets) 

[Open Scivis Datasets (SCI Backup)](http://open-scivis-datasets.sci.utah.edu/open-scivis-datasets/)


## Setup
Run the following scripts to generate the HTML files and download datasets. You can configure the url of your website in the `config.json` file.

```
py generate_website.py
py download_raw_files.py
```

To test the website locally run `py server.py`, and the website is accessible at `http://localhost:8000/open-scivis-datasets`. This Python server
is suitable only for testing because large file downloads can run out of memory. Furthermore, the server does not support HTTPS connections. Therefore, use the `server.go` which supports streaming and HTTPS.


## Adding New Dataset

New dataset is added by following these steps:

1. Call `python add_dataset_from_raw.py` script. It will create a directory in `static/open-scivis-datasets` folder, copy the raw file, create metadata JSON file, and preview.
2. Fill in the rest of details in the JSON metadata file.
3. Add link to `test.py`.


## Version 2.0
- Value range (or show example how to use Python to compute them)
- Histogram (Python sample to compute them?)
- Volume rendering (https://arxiv.org/abs/2406.15634)
- render multiple surfaces (+ opacity)?
- improve quality of previews (rendering)
- render continuously instead of firing on events. Should make it smoother. (what about battery life?)
- unicode in bibtex?
- make it easy to run locally as simple repository for user's personal use (can organize datasets, get quick previews); generate json from raw file name
- performance benchmark for OpenVisus queries
- isovalue slider is hard to use for data that have nonlinear scale
- Box selection with clipping planes, click on box sides to shrink
- highlight datasets with BibTex, on each dataset title have a link or icon to bibtex
- fields (with previews)
- versioning of datasets.json file
- Better description of the data, what would one expect in the data? If one is looking for something. Speed of the shockwave, who simulation it is from. Help people find data they want.
- Show slices of data in the preview.
- Date when dataset was added (and produced?), new datasets are more visible
- reverse references to open scivis (list all papers that use a particular dataset. How to do it? (use Google form to allow authors report their papers)
- JICF vector field (we now share only the Q criterion); time dependent dataset
- automatically choose reasonable isovalue
