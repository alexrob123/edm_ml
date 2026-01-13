# EDM Multi-Label

Working on conditional generation evaluation with Multi-Label prediction of condition.

This illustrative example is run with CelebA dataset and selected attributes.

## Dataset

The project is based on [CelebA aligned and cropped dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

For automatic download, you can update and run:

```bash
uv run download.py --data-dir ~/data
cd ~/data/CelebA64
unzip img_align_celeba.zip
```

- Create dir `~/data/CelebA64`
- Download `img_align_celeba.zip` and `list_attr_celeba.txt`
- Unzip `img_align_celeba.zip`

Or download dataset by hand [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Subset preparation

Subdata folders per attribute combinations can be generated with default:

```bash
uv run build_subsets.py \
    --data-dir ~/data/CelebA64 \
    --img-dir img_align_celeba \
    --attr-file list_attr_celeba.txt \
    --out-dir _combinations \
    --attrs Bangs Eyeglasses Male Smiling
```

With default parameters, data directory should look like:

```
~/data/CelebA64
├── \_combinations
├── img_align_celeba
├── img_align_celeba.zip
└── list_attr_celeba.txt
```

```
~/data/CelebA64/_combinations
├── Bangs
├── Bangs_Eyeglasses
├── Bangs_Eyeglasses_Male
├── Bangs_Eyeglasses_Male_Smiling
├── Bangs_Eyeglasses_Smiling
├── Bangs_Male
├── Bangs_Male_Smiling
├── Bangs_Smiling
├── Eyeglasses
├── Eyeglasses_Male
├── Eyeglasses_Male_Smiling
├── Eyeglasses_Smiling
├── Male
├── Male_Smiling
├── Smiling
├── subset_counts.txt
└── subset_imgs.txt
```
