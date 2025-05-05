path_to_config="configs/test_interpolation_3_prompts.yaml"
CUDA_VISIBLE_DEVICES=0 python launch.py --config ${path_to_config} --train --gpu 0 \
