path_to_config="configs/test_interpolation.yaml"
python launch.py --config ${path_to_config} --export --gpu 0 \
resume=<path_to_weights>
