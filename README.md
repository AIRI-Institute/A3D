https://voyleg.github.io/a3d/vid/motivation/motivation.mp4


#  A3D: Does Diffusion Dream about 3D Alignment?
Repository for the generation of the aligned 3D meshes, or editing the existing meshes in the consistent way.  
[Project Page](https://voyleg.github.io/a3d/) | [Paper](https://arxiv.org/abs/2406.15020)

## Preparation
Clone the A3D repository  
  ```
  git clone https://github.com/AIRI-Institute/A3D --recurse-submodules
  ```
## Installation

### Docker image
We provide the docker image based on the Richdreamer image [Richdreamer](https://github.com/modelscope/richdreamer).  
Our docker image can be found here [here](https://hub.docker.com/r/savvai/mvdream_richdreamer).  
- Log into your dockerhub accout  
  ```
  docker login
  ```
- Download Docker image  
  ```
  docker pull savvai/mvdream_richdreamer:latest
  ```
- Verify the download  
  ```
  docker images
  ```
- Move to the project folder
  ```
  cd morphology-inr
  ```
- Run the image  
  ```
  docker run -it --gpus all -v /absolute/path/to/dir/morphology-inr/:/morphology-inr/ savvai/mvdream_richdreamer:latest
  ```

### Dockerfile
We also provide an updated dockerfile from [repository](https://github.com/bytedance/MVDream), since the old dockerfile and requiriments are outdated.
- Move to the docker folder
  ```
  cd morphology-inr/MVDream-threestudio/docker
  ```
- Build docker image from dockerfile
  ```
  docker build -t savvai/mvdream:latest .
  ```
- Move back to the project folder and change rights
  ```
  cd .. && cd ..
  chmod 777 -R .
  ```
- Run the image
  ```
  docker run -it --gpus all -v /absolute/path/to/dir/morphology-inr/:/morphology-inr/ savvai/mvdream:latest
  ```

## Quickstart 

Move into MVDream backbone folder 
  ```
  cd /morphology-inr/MVDream-threestudio
  ```

### Pair generation
For generating the pairs of the objects we are providing the prepared config file. You can manually specify the prompts through the command line. Choose the pair of the objects with analogous geometry and run the following command.  
``` 
python launch.py --config configs/mvdream-interpolation-sd21-shading.yaml --train system.prompt_processor_A.prompt="dog animal"  system.prompt_processor_B.prompt="cat animal"
```  
Where you should replace "dog animal" and "cat animal" with the prompts of your choice.  

### Multi-prompt generation
If you want to generate more than two objects you need to switch to the multi-prompt hybridization branch:  
  ```
  git fetch
  git checkout multiprompt_hybridization
  ```  

#### Training

To run training:

```
bash scripts/train.sh
```

In `train.sh` file you have to specify `path_to_config`.

In config there are several points, that should be considered:

- `exp_root_dir` -- dir for saveing results

- `n_prompts` -- number of processed prompts

- `n_parameter_dims` -- number of dim for interpolation

- prompts in `prompt_processor_list`

Be aware that `n_prompts == n_parameter_dims == len(prompt_processor_list)`.


#### Test
To run test:

```
bash scripts/test.sh
```

In `test.sh` file you have to specify `path_to_config` and `--resume` weights.

For gibridization run you should specify in `renderer` value `interpolation`:

```
interpolation:
      anchors:
        point_1: [0.5, 0.00, 0.25]
        point_2: [0.3, -0.0, 0.4]
        point_3: [0.3, -0.0, 0.2]
        point_4: [-0.15, 0.0, -0.1]
       
        
      vals:
        point_1: [1, 0]
        point_2: [1, 0]
        point_3: [0, 1]
        point_4: [0, 1]
```

where `anchors` characterize the anchor points coordinates in 3D space of mesh and `vals` corresponds to chosen number of prompt.

- Then interpolation based on anchors in applyed to the whole 3D space to produce final hybridization mesh.

- If no parameters are provided -- the space will be devided in a half and hybridization will be applied based on such division.

- As the result you will obtain images for all prompts and for hybridization.

- In case of two prompts you will also obtain transition between theese prompts.


#### Export

To run export:

```
bash scripts/export.sh
```

In `export.sh` file you have to specify `path_to_config` and `--resume` weights.

During export you will obtain mesh for all prompts and for hibridization object.

## Tips
- **Try to use concrete prompts**.
   
  Due to some bias MVDream sometimes produce unintended results. Thus, try to use more concrete prompts, like "cat animal" instead of "cat".
- **Try to use structurally related prompts to improve alignment**.
  
  Sometimes it is better to use similar prompts in order to hint the diffusion model what you want to get as a result. Using shared word "cat animal" and "dog animal" prompts in some cases could obtain better results that "cat" and "dog".
- **Add the ", no background" suffix to the prompt**.
  
  If the experiments tend to fall with "CUDA out of memory" error. It is a known issue with the MVDream pipeline connected with the nerfacc.

## Citing 
Please cite our article in the following way:
```
@inproceedings{ignatyev2025a3d,
    title      = {{A3D}: Does Diffusion Dream about 3D Alignment?},
    shorttitle = {{A3D}},
    author     = {Ignatyev, Savva and Konovalova, Nina and Selikhanovych, Daniil and Voynov, Oleg and Patakin, Nikolay and Olkov, Ilya and Senushkin, Dmitry and Artemov, Alexey and Konushin, Anton and Filippov, Alexander and Wonka, Peter and Burnaev, Evgeny},
    year       = {2025},
    booktitle  = {The Thirteenth International Conference on Learning Representations},
    url        = {https://openreview.net/forum?id=QQCIfkhGIq},
    langid     = {english}
}
```

## Acknowledgements

This repository is based on [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio), developed by Yifan Jiang, Shuai Yang, Peihao Zhu, Yujun Shen, Yujiao Chen, Xintao Wang, Yingyan Lin, and Bolei Zhou. The original code is licensed under the Apache License, Version 2.0.

## Modifications

The following modifications have been made to the original [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio) codebase:

### main:
- Added the parameterised version of the implicit volume geometry backbone, which supports the modelling of the continuous transitions between the pairs of objects. 
- Added the interpolation-based version of the SDS diffusion guidance for the intermediate guidance of the transition samples for the pairs of the objects.
- Added the version of the renderer which supports rendering the intermediate steps of the transition between the objects
- Augmented the data sampler to sample the latent variable.
- Modified the mesh exporter for the consistency with the modified code.

### multiprompt_hybridization:
- Added the parameterised version of the implicit volume geometry backbone, which supports the modelling of the continuous transitions for multi-object generation. 
- Added the interpolation-based version of the SDS diffusion guidance for the intermediate guidance of the transition samples for the multiple objects.
- Added the version of the renderer which supports rendering the intermediate steps of the transition between the multiple objects
- Augmented the data sampler to sample the latent variable from a high-dimension space.
- Added the possibility to interpolate 3D space between provided prompts during inference, based on chosen anchor points and corresponding prompts for hybridization task.
- Added the possibility to export the interpolated (hybridized) mesh based on provided anchor points and corresponding prompts.

