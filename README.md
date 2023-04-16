All scripts and metadata necessary to deploy the [Open Scivis Datasets](https://klacansky.com/open-scivis-datasets) website. If you are not on Windows, you will need to
rebuild [zfp](https://github.com/LLNL/zfp) library for your platform of choice. Scripts to create
[OpenVisus](https://github.com/sci-visus/OpenVisus) IDX files are missing because this feature is experimental.


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

1. Create a new directory in `static/open-scivis-datasets`. The directory name is used as the dataset identifier, enforcing a distinct name per dataset.
2. Compute SHA 512 checksum. On Windows, use `certutil -hashfile file.raw SHA512`.
3. Copy and fill the `static/open-scivis-datasets/template.json` metadata file.
4. Generate preview using the `generate_preview.py` script.
5. Add link to `test.py`.


## Version 2.0
- unicode in bibtex?
- make it easy to run locally as simple repository for user's personal use (can organize datasets, get quick previews); generate json from raw file name
- help user to compute sha512 checksum automatically?
- should we store previews in Github? (or generate them all at once at deployment time? the website is useful with previews only)
- new previews should be <= 512x512x512 voxels; increase zfp precision too?
- performance benchmark for OpenVisus queries
- compile ZFP into wasm so precision argument can be selected
- isovalue slider is hard to use for data that have nonlinear scale
- Box selection with clipping planes, click on box sides to shrink
- sort in descending order by size and number of voxels? (but alphabetical is ascending)
- highlight datasets with BibTex, on each dataset title have a link or icon to bibtex
- fields (with previews)
- versioning of datasets.json file
- keep old links working by using redirects (or symbolic links); keep old datasets.json there
- Better description of the data, what would one expect in the data? If one is looking for something. Speed of the shockwave, who simulation it is from. Help people find data they want.
- Show slices of data in the preview.
- Date when dataset was added (and produced?), new datasets are more visible
- reverse references to open scivis (list all papers that use a particular dataset. How to do it? (use Google form to allow authors report their papers)
- JICF vector field (we now share only the Q criterion); time dependent dataset
- opening link directly with preview sometimes causes JS error (visible on console); likely related to compiled zfp to wasm https://klacansky.com/open-scivis-datasets/#hcci_oh
- rename spacing to voxel_scaling?
- reduced precision/resolution box queries using IDX2
- automatically choose reasonable isovalue
- show isovalue selected in the preview (as text) (if preview is used in paper, people can see the isovalue; also show resolution and precision)
- include value range for each dataset (min and max)?
- If one preview is being downloaded and the user selects another preview, it may show the old preview (it will also jump the % progress display between the two)
