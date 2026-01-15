# EDM Multi-Label

Working on conditional generation evaluation with Multi-Label prediction of condition.

This illustrative example is run with CelebA dataset and selected attributes.

## Dataset

The project is based on [CelebA aligned and cropped dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

For automatic download, you can update and run:

```bash
uv run download.py --dir ~/data --aligned 1 --extension jpg
cd ~/data/CelebA/AlignedCropped/JPG
unzip img_align_celeba.zip
```

- Create dir `~/data/CelebA/AlignedCropped/JPG`
- Download `img_align_celeba.zip` and `list_attr_celeba.txt`
- Unzip `img_align_celeba.zip`

Or download dataset by hand [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Subset preparation

Dataset is build to the be compatible with [EDM](https://github.com/NVlabs/edm) project, that is in the [StyleGAN](https://github.com/NVlabs/stylegan3) format. For a given attributes selection, the dataset is split in non-overlapping subsets per attribute combination, mimicking a Label Powerset approach from a Multi-Label setup.

Subdata folders per attribute combinations can be generated with default:

```bash
uv run build_subsets.py \
    --data-dir ~/data/CelebA/AlignedCropped/JPG \
    --img-dir img_align_celeba \
    --attr-file list_attr_celeba.txt \
    --out-dir _NonOverlappingClasses \
    --attrs Bangs Eyeglasses Male Smiling
```

With default parameters, data directory should look like:

```
~/data/CelebA/AlignedCropped/JPG
├── \_NonOverlappingClasses
├── img_align_celeba
├── img_align_celeba.zip
└── list_attr_celeba.txt
```

```
~/data/CelebA/AlignedCropped/JPG/_NonOverlappingClasses
    ├── selection_hash_dir
    │   ├── combination_1_hash_dir
    │   │   ├── 000003.jpg
    │   │   └── 000007.jpg
    │   ├── combination_2_hash_dir
    │   │   ├── 000001.jpg
    │   │   └── 000002.jpg
    │   ├── ...
    │   └── metadata.json
```

Root directory can then be fed to [EDM](https://github.com/NVlabs/edm) `dataset_tool.py` see dedicated section [Preparing dataset](https://github.com/NVlabs/edm/blob/main/docs/dataset-tool-help.txt) and [`python dataset_tool.py --help`](https://github.com/NVlabs/edm/blob/main/docs/dataset-tool-help.txt).
