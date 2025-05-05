path_to_config="configs/test_interpolation.yaml"
CUDA_VISIBLE_DEVICES=0 python launch.py --config ${path_to_config} --test --gpu 0 \
resume=<path_to_weights>
