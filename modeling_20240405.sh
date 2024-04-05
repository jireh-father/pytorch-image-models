CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_style_name --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 7 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_style_name > log_tr_efb0_hair_style_name.log &

# curl type
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_type --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 6 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_curl_type > log_tr_efb0_curl_type.log &

# curl type merged c
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/curl_type_merge_c --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 5 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_curl_type_merged_c > log_tr_efb0_curl_type_merged_c.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_length --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 4 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_length > log_tr_efb0_hair_length.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/bangs --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 4 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_bangs > log_tr_efb0_bangs.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/cut --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 2 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_cut > log_tr_efb0_cut.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --data-dir /source/pytorch-image-models/dataset/labeled_v1.3/hair_thickness --dataset ImageFolder --model efficientnet_b0 --pretrained \
 --num-classes 2 --img-size 224 --batch-size 128 --validation-batch-size 128 --epochs 100 --log-interval 100 --output ./output/efficientnet_b0_hair_thickness > log_tr_efb0_hair_thickness.log &

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