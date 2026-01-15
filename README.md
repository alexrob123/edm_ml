# EDM Multi-Label

This repo explores conditional generative model evaluation with Multi-Label prediction.
This illustrative example is run with CelebA dataset and a selection of available attributes.

## Dataset

At the moment the project revolves around [CelebA aligned and cropped dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Other datasets might be added later.

For automatic download, you can update and run:

```bash
uv run download.py --dir ~/data --aligned 1 --extension jpg
cd ~/data/CelebA/AlignedCropped-JPG
unzip img_align_celeba.zip
```

- Create dir `~/data/CelebA/AlignedCropped-JPG`
- Download `img_align_celeba.zip` and `list_attr_celeba.txt`
- Unzip `img_align_celeba.zip`

Or download dataset by hand [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Powerset partitions

For a given attributes selection, the original dataset is split in non-overlapping subsets per attribute combination (labelsets in the powerset of the selection, cf multi-label learning) and stored in corresponding folder. Each folder contains the images corresponding the attribute combination.

Dataset is build to the be compatible with [EDM](https://github.com/NVlabs/edm) project, that is in the [StyleGAN](https://github.com/NVlabs/stylegan3) format.

Subdata folders per labelset in the selected attributes powerset can be generated with default:

```bash
uv run build_subsets.py \
    --data-dir ~/data/CelebA/AlignedCropped-JPG \
    --img-dir img_align_celeba \
    --attr-file list_attr_celeba.txt \
    --out-dir _powerset_partitions \
    --attrs Bangs Eyeglasses Male Smiling
```

With default parameters, data directory should look like:

```
~/data/CelebA/AlignedCropped-JPG
├── \_powerset_partitions
├── img_align_celeba
├── img_align_celeba.zip
└── list_attr_celeba.txt
```

```
~/data/CelebA/AlignedCropped-JPG/_powerset_partitions
    ├── selection_hash
    │   ├── combination_1_hash
    │   │   ├── 000003.jpg
    │   │   └── 000007.jpg
    │   ├── combination_2_hash
    │   │   ├── 000001.jpg
    │   │   └── 000002.jpg
    │   ├── ...
    │   └── metadata.json
```

Newly created `_powerset_partitions/selection_hash` directory can then be fed to the [EDM](https://github.com/NVlabs/edm) `dataset_tool.py` added to the project. For more information on the tool, see dedicated section [Preparing dataset](https://github.com/NVlabs/edm/blob/main/docs/dataset-tool-help.txt) and [`python dataset_tool.py --help`](https://github.com/NVlabs/edm/blob/main/docs/dataset-tool-help.txt).

```bash
uv run dataset_tool.py \
    --source ~/data/CelebA/AlignedCropped-JPG/_powerset_partitions/<selection_hash> \
    --dest  ~/data/CelebA/edm-64x64/<selection_hash>.zip \
    --transform center-crop \
    --resolution 64x64
```
