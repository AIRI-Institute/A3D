export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"

path_to_config="/home/jovyan/users/konovalova/workspace/morphology-inr/MVDream-threestudio/configs/config_test/mvdream_other_cactus.yaml"
CUDA_VISIBLE_DEVICES=0 python launch.py --config ${path_to_config} --train --gpu 0 \
#resume="outputs_other_interpolation/mvdream-sd21-rescale0.5-shading/cactus_plant@20240510-184704/ckpts/last.ckpt"
 