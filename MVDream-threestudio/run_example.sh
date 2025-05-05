path_to_config="configs/mvdream-interpolation-sd21-shading.yaml"
CUDA_VISIBLE_DEVICES=7 python launch.py --config ${path_to_config} --train --gpu 0 \
system.prompt_processor_A.prompt="horse animal" \
system.prompt_processor_B.prompt="horse skeleton" \
system.exporter_type="mesh-exporter-mvdream-interpolation" \
system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256 \
system.exporter.context_type="cuda" \
system.exporter.fmt='obj' \
system.exporter.save_uv=False \
system.exporter.save_texture=False
