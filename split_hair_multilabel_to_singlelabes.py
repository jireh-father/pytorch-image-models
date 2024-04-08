import argparse
import random
import os
import shutil
import glob
import json
from collections import defaultdict

hair_color_list = {
    'gray hair': True,
    'purple hair': True,
    'brown hair': True,
    'black hair': True,
    'blonde hair': True,
    'gold brown hair': True,
    'dark brown hair': True,
    'red violet hair': True,
    'red brown hair': True,
    'red orange hair': True,
    'red wine hair': True,
    'matte brown hair': True,
    'mushroom blonde hair': True,
    'milk brown hair': True,
    'ash gray hair': True,
    'ash violet hair': True,
    'ash beige hair': True,
    'ash brown hair': True,
    'ash blonde hair': True,
    'ash blue hair': True,
    'ash khaki hair': True,
    'ash khaki brown hair': True,
    'ash purple hair': True,
    'ash pink hair': True,
    'orange brown hair': True,
    'choco brown hair': True,
    'khaki brown hair': True,
    'pink brown hair': True,
    'red hair': True,
    'multicolored hair': True,
    'grey hair': True,
    'pink hair': True,
    'blue hair': True,
    'orange hair': True,
    'white hair': True,
    'gradient hair': True,
    'two-tone hair': True,
    'green hair': True,
}

val_to_key_map = {
    "bob hair": "hair_length",
    "short hair": "hair_length",
    "medium hair": "hair_length",
    "long hair": "hair_length",
    "no-curl": "curl_type",
    "c-curl perm": "curl_type",
    "s-curl perm": "curl_type",
    "s3-curl perm": "curl_type",
    "inner c-curl perm": "curl_type",
    "outer c-curl perm": "curl_type",
    "cs-curl perm": "curl_type",
    "ss-curl perm": "curl_type",
    "c-shaped curl perm": "curl_type",
    "s-shaped curl perm": "curl_type",
    "s3-shaped curl perm": "curl_type",
    "inner c-shaped curl perm": "curl_type",
    "outer c-shaped curl perm": "curl_type",
    "cs-shaped curl perm": "curl_type",
    "ss-shaped curl perm": "curl_type",
    "twist curl perm": "curl_type",
    "thin curl": "curl_width",
    "thick curl": "curl_width",
    "no-layered hair": "cut",
    "layered hair": "cut",
    "hush cut": "hair_style_name",
    "tassel cut": "hair_style_name",
    "hug perm": "hair_style_name",
    "build perm": "hair_style_name",
    "slick cut": "hair_style_name",
    "short cut": "hair_style_name",
    "layered cut": "hair_style_name",
    "hippie perm": "hair_style_name",
    "full bangs": "bangs",
    "side bangs": "bangs",
    "see-through bangs": "bangs",
    # "choppy bangs": "bangs",
    "faceline bangs": "bangs",
    "thin hair": "hair_thickness",
    "thick hair": "hair_thickness",
    "9:1 hair part": "hair_part",
    "8:2 hair part": "hair_part",
    "7:3 hair part": "hair_part",
    "5:5 hair part": "hair_part",
    "3:7 hair part": "hair_part",
    "2:8 hair part": "hair_part",
    "1:9 hair part": "hair_part",
    'gray hair': 'hair_color',
    # 'purple hair': 'hair_color',
    'brown hair': 'hair_color',
    'black hair': 'hair_color',
    'blonde hair': 'hair_color',
    'gold brown hair': 'hair_color',
    'dark brown hair': 'hair_color',
    # 'red violet hair': 'hair_color',
    'red brown hair': 'hair_color',
    # 'red orange hair': 'hair_color',
    # 'red wine hair': 'hair_color',
    'matte brown hair': 'hair_color',
    # 'mushroom blonde hair': 'hair_color',
    'milk brown hair': 'hair_color',
    'ash gray hair': 'hair_color',
    # 'ash violet hair': 'hair_color',
    'ash beige hair': 'hair_color',
    'ash brown hair': 'hair_color',
    # 'ash blonde hair': 'hair_color',
    'ash blue hair': 'hair_color',
    # 'ash khaki hair': 'hair_color',
    # 'ash khaki brown hair': 'hair_color',
    # 'ash purple hair': 'hair_color',
    # 'ash pink hair': 'hair_color',
    'orange brown hair': 'hair_color',
    'choco brown hair': 'hair_color',
    # 'khaki brown hair': 'hair_color',
    'pink brown hair': 'hair_color',
    # 'red hair': 'hair_color',
    # 'multicolored hair': 'hair_color',
    'grey hair': 'hair_color',
    # 'pink hair': 'hair_color',
    # 'blue hair': 'hair_color',
    # 'orange hair': 'hair_color',
    # 'white hair': 'hair_color',
    # 'gradient hair': 'hair_color',
    # 'two-tone hair': 'hair_color',
    'green hair': 'hair_color',
}

merge_class_map = {
    "9:1 hair part": "7:3 hair part",
    "8:2 hair part": "7:3 hair part",
    "1:9 hair part": "3:7 hair part",
    "2:8 hair part": "3:7 hair part",
    "s3-curl perm": "s-curl perm",
    "ss-curl perm": "s-curl perm",
    # "mushroom blonde hair": "blonde hair",
    # "ash khaki brown hair": "khaki brown hair",
    "grey hair": "gray hair"

}


def split_tags(tag_str, merge_c_curl):
    tags = []
    for tag in tag_str.split(", "):
        if merge_c_curl:
            if "c-shaped curl perm" in tag or "c-curl perm" in tag:
                tags.append("c-curl perm")
                continue

        if tag in val_to_key_map:
            tags.append(tag)

    return tags


def main(args):
    if args.random_seed is not None:
        random.seed(args.random_seed)

    label_data = json.load(open(args.label_file, encoding='utf-8'))

    tag_set = set()
    tag_stat = defaultdict(int)
    task_dict = defaultdict(dict)
    for file_name in label_data:
        tags = split_tags(label_data[file_name]["tags"], args.merge_c_curl)
        tag_set.update(tags)
        for tag in tags:
            if tag in merge_class_map:
                tag = merge_class_map[tag]
            tag_stat[tag] += 1
            if tag in task_dict[val_to_key_map[tag]]:
                task_dict[val_to_key_map[tag]][tag].append(f"{file_name}.jpg")
            else:
                task_dict[val_to_key_map[tag]][tag] = [f"{file_name}.jpg"]

    for task_name in task_dict:
        print("task_name", task_name, len(task_dict[task_name]))
        class_dict = task_dict[task_name]
        # class_dict format = {"class_name1": ["file_name1.jpg", "file_name2.jpg", ...], ...}
        # split class_dict into train and val, each class val has 10%, shuffle
        for class_name in class_dict:
            print("class_name", class_name, len(class_dict[class_name]))
            random.shuffle(class_dict[class_name])
            num_train = int(len(class_dict[class_name]) * 0.9)
            train_class_dict = class_dict[class_name][:num_train]
            val_class_dict = class_dict[class_name][num_train:]

            train_class_dir = os.path.join(args.output_dir, task_name, "train", class_name)
            val_class_dir = os.path.join(args.output_dir, task_name, "validation", class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            # move val_class_dict to val_dir
            for file_name in val_class_dict:
                src_file = os.path.join(args.image_dir, file_name)
                shutil.copy(src_file, val_class_dir)

            # move train_class_dict to train_dir
            for file_name in train_class_dict:
                src_file = os.path.join(args.image_dir, file_name)
                shutil.copy(src_file, train_class_dir)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_file', type=str, default='')
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    # random_seed
    parser.add_argument('--random_seed', type=int, default=0)
    # merge_c_curl
    parser.add_argument('--merge_c_curl', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/ml_to_sl/bangs --dataset ImageFolder --model efficientnet_b0 --pretrained \
# --num-classes 4 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_bangs > log_tr_efb0_bangs.log &

# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/ml_to_sl/hair_length --dataset ImageFolder --model efficientnet_b0 --pretrained \
# --num-classes 4 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_length > log_tr_efb0_hair_length.log &

# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/ml_to_sl/hair_style_name --dataset ImageFolder --model efficientnet_b0 --pretrained \
# --num-classes 7 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_style_name > log_tr_efb0_hair_style_name.log &

# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/ml_to_sl/hair_thickness --dataset ImageFolder --model efficientnet_b0 --pretrained \
# --num-classes 2 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_thickness > log_tr_efb0_hair_thickness.log &

# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/ml_to_sl/curl_type --dataset ImageFolder --model efficientnet_b0 --pretrained \
# --num-classes 4 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_curl_type > log_tr_efb0_curl_type.log &

# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/ml_to_sl/cut --dataset ImageFolder --model efficientnet_b0 --pretrained \
# --num-classes 2 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_cut > log_tr_efb0_cut.log &
