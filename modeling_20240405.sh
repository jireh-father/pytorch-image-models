CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_style_name --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 7 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_style_name --eval-metric f1 > log_tr_efb0_hair_style_name.log &

# batch 4 > batch 100ㅇㅣ 약간 더 좋음
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_style_name --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 7 --img-size 224 --batch-size 4 --validation-batch-size 4 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_style_name_batch4 --eval-metric f1 > log_tr_efb0_hair_style_name_batch4.log &

# aug test & batch 32
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_style_name --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 8 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_style_name --eval-metric f1 \
 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 --scale 0.65 1.0 --ratio 0.95 1.05 > log_tr_efb0_hair_style_name.log &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_type --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 6 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_curl_type --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_curl_type.log &

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_part --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 3 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_part --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_hair_part.log &

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_length --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 4 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_length --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_hair_length.log &

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/bangs --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 4 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_bangs --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_bangs.log &

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/cut --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 2 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_cut --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_cut.log &

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_thickness --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 2 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_thickness --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_hair_thickness.log &

# curl_width
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_width --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 2 --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_curl_width --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_curl_width.log &

# hair_color
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_color --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 16 --use-class-weights --img-size 224 --batch-size 32 --validation-batch-size 32 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_color_cweight --eval-metric f1 --cutmix 0.4 --mixup 0.5 --drop 0.1 --crop-pct 1.0 > log_tr_efb0_hair_color_cweight.log &

# inference
CUDA_VISIBLE_DEVICES=2 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 8 --checkpoint ./output/efficientnet_b0_hair_style_name/20240406-140949-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_hair_style_name --include-index > log_infer_hair_style_name &
CUDA_VISIBLE_DEVICES=2 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 6 --checkpoint ./output/efficientnet_b0_curl_type/20240406-140949-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_curl_type --include-index > log_infer_curl_type &
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 4 --checkpoint ./output/efficientnet_b0_hair_length/20240405-145842-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_hair_length --include-index > log_infer_hair_length &
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 4 --checkpoint ./output/efficientnet_b0_bangs/20240405-145843-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_bangs --include-index > log_infer_bangs &
CUDA_VISIBLE_DEVICES=2 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 2 --checkpoint ./output/efficientnet_b0_cut/20240405-191529-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_cut --include-index > log_infer_cut &
CUDA_VISIBLE_DEVICES=2 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 2 --checkpoint ./output/efficientnet_b0_hair_thickness/20240405-191530-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_hair_thickness --include-index > log_infer_hair_thickness &
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 2 --checkpoint ./output/efficientnet_b0_curl_width/20240406-134138-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_curl_width --include-index > log_infer_curl_width &

CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 3 --checkpoint ./output/efficientnet_b0_hair_part/20240405-145842-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_hair_part --include-index > log_infer_hair_part &

# hair_color inference
CUDA_VISIBLE_DEVICES=1 nohup python -u inference.py --data-dir ./dataset/unlabeled --dataset ImageFolder --model efficientnet_b0 --img-size 224 --crop-pct 1.0 --num-classes 16 --checkpoint ./output/efficientnet_b0_hair_color/20240408-103139-efficientnet_b0-224/model_best.pth.tar  --results-dir ./infer_results --results-format json --results-file infer_hair_color --include-index > log_infer_hair_color &


# b3
# hair_style_name
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_style_name --dataset ImageFolder --model efficientnet_b3 --pretrained \
 --num-classes 7 --batch-size 64 --validation-batch-size 64 --epochs 100 --log-interval 100 --output ./output/efficientnet_b3_hair_style_name > log_tr_efb3_hair_style_name.log &
# hair_length
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_length --dataset ImageFolder --model efficientnet_b3 --pretrained \
 --num-classes 4 --batch-size 64 --validation-batch-size 64 --epochs 100 --log-interval 100 --output ./output/efficientnet_b3_hair_length > log_tr_efb3_hair_length.log &

# curl type
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_type --dataset ImageFolder --model efficientnet_b3 --pretrained \
 --num-classes 6 --batch-size 64 --validation-batch-size 64 --epochs 100 --log-interval 100 --output ./output/efficientnet_b3_curl_type > log_tr_efb3_curl_type.log &

# curl type merged c
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_type_merge_c --dataset ImageFolder --model efficientnet_b3 --pretrained \
 --num-classes 5 --batch-size 64 --validation-batch-size 64 --epochs 100 --log-interval 100 --output ./output/efficientnet_b3_curl_type_merged_c > log_tr_efb3_curl_type_merged_c.log &

# bangs
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/bangs --dataset ImageFolder --model efficientnet_b3 --pretrained \
 --num-classes 4 --batch-size 64 --validation-batch-size 64 --epochs 100 --log-interval 100 --output ./output/efficientnet_b3_bangs > log_tr_efb3_bangs.log &

# cut
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/cut --dataset ImageFolder --model efficientnet_b3 --pretrained \
 --num-classes 2 --batch-size 64 --validation-batch-size 64 --epochs 100 --log-interval 100 --output ./output/efficientnet_b3_cut > log_tr_efb3_cut.log &

# hair_thickness
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_thickness --dataset ImageFolder --model efficientnet_b3 --pretrained \
 --num-classes 2 --batch-size 64 --validation-batch-size 64 --epochs 100 --log-interval 100 --output ./output/efficientnet_b3_hair_thickness > log_tr_efb3_hair_thickness.log &





# vis
# hair_style_name
CUDA_VISIBLE_DEVICES=2 nohup python -u timm_grad_cam.py --image_dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_style_name/validation/ --model_path /source/pytorch-image-models/output/efficientnet_b0_hair_style_name/20240222-135441-efficientnet_b0-224/model_best.pth.tar --output_dir /source/pytorch-image-models/vis/hair_style_name_b0 > log_vis_hair_style_name_b0.log &
# hair_length
CUDA_VISIBLE_DEVICES=2 nohup python -u timm_grad_cam.py --image_dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_length/validation/ --model_path /source/pytorch-image-models/output/efficientnet_b0_hair_length/20240222-145951-efficientnet_b0-224/model_best.pth.tar --output_dir /source/pytorch-image-models/vis/hair_length_b0 > log_vis_hair_length_b0.log &

# curl_type
CUDA_VISIBLE_DEVICES=2 nohup python -u timm_grad_cam.py --image_dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_type/validation/ --model_path /source/pytorch-image-models/output/efficientnet_b0_curl_type/20240222-141336-efficientnet_b0-224/model_best.pth.tar --output_dir /source/pytorch-image-models/vis/curl_type_b0 > log_vis_curl_type_b0.log &
# curl_type_merged_c
CUDA_VISIBLE_DEVICES=2 nohup python -u timm_grad_cam.py --image_dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_type_merge_c/validation/ --model_path /source/pytorch-image-models/output/efficientnet_b0_curl_type_merged_c/20240222-142706-efficientnet_b0-224/model_best.pth.tar --output_dir /source/pytorch-image-models/vis/curl_type_merged_c_b0 > log_vis_curl_type_merged_c_b0.log &

# bangs
CUDA_VISIBLE_DEVICES=2 nohup python -u timm_grad_cam.py --image_dir /source/pytorch-image-models/dataset/labeled_v1.3/bangs/validation/ --model_path /source/pytorch-image-models/output/efficientnet_b0_bangs/20240222-152000-efficientnet_b0-224/model_best.pth.tar --output_dir /source/pytorch-image-models/vis/bangs_b0 > log_vis_bangs_b0.log &
# cut
CUDA_VISIBLE_DEVICES=2 nohup python -u timm_grad_cam.py --image_dir /source/pytorch-image-models/dataset/labeled_v1.3/cut/validation/ --model_path /source/pytorch-image-models/output/efficientnet_b0_cut/20240222-153144-efficientnet_b0-224/model_best.pth.tar --output_dir /source/pytorch-image-models/vis/cut_b0 > log_vis_cut_b0.log &
# hair_thickness
CUDA_VISIBLE_DEVICES=2 nohup python -u timm_grad_cam.py --image_dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_thickness/validation/ --model_path /source/pytorch-image-models/output/efficientnet_b0_hair_thickness/20240222-154402-efficientnet_b0-224/model_best.pth.tar --output_dir /source/pytorch-image-models/vis/hair_thickness_b0 > log_vis_hair_thickness_b0.log &