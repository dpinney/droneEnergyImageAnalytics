# Pole Detection

NRECA Pole detection project initial stage, by James Zhou

## Part 1 - Synthetic Image Generation

Since we do not have any real data available yet, I decided to make a synthetic image dataset

### Prerequisites

This program need python 3 and following packages:

```
json
random
numpy
pathlib
tqdm
pillow
scikit-image
shapely
```
Picture Requirements:

1. RGB format
2. Exif, XMP, IPTC data have to be removed


## Running the program

1. Run `image_composition.py` to create synthetic datasets, `dataset_info.json` and `mask_definitioins.json`

 Expected image directory structure:
 
 input_image_dir
    + foregrounds_dir
        + super_category_dir
            + category_dir
                    foreground_image.png
    + backgrounds_dir
            back_image.png
        

Here is an bash example of creating 20 images on my machine: 
```
python ./python/image_composition.py --input_dir ./datasets/input --output_dir ./datasets/output --count 20 --width 512 --height 512
```
2.  Run `coco_json_utils.py` to create `coco_instances.json` from  `dataset_info.json` and `mask_definitioins.json`.

Bash Eample:

```
python ./python/coco_json_utils.py -md ./datasets/output/mask_definitions.json -di ./datasets/output/dataset_info.json
```

## Part 2 - Train and test Mask-R-CNN

To be continued
