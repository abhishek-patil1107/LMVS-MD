# LMVS-MD

## NOTE: Still developing the learning algorithm
Learning MVS-Quality Depth from Monocular Estimation

No existing method learns to elevate monocular depth predictions to multi-view stereo quality in a feed-forward, generalizable manner.
Monocular depth errors are systematic and scene-invariant, thus learnable.

Examples of learnable patterns:

1. Sky always at infinite depth
2. Floors are planar
3. Depth compression at distance follows power law
4. Edge sharpness correlates with image gradients

Perhaps by having algorithms learn these patterns, we can lower the reconstruction time by a lot.



Applications/Software used: 
1. Meshlab: For editing vertices, filling in meshes, reorienting meshes.
2. Blender: Flatten vertices a the base to ensure stable grasping.
3. Spyder: Python code development IDE

repositories used:
1. XM-Code - https://github.com/ComputationalRobotics/XM-code
2. Colmap - https://github.com/colmap/colmap
3. Simple-MuJoCo-PickNPlace - https://github.com/volunt4s/Simple-MuJoCo-PickNPlace


## Steps taken:

1. Create python environment
2. Install requirements
3. Perform 3D reconstruction
4. Import reconstruction to Meshlab
5. Edit meshes in Meshlab
6. Flatten bottom surface in Blender
7. Import mesh into MuJoCo simulation for Pick and Place operation

![Example image from the mip-Nerf360 dataset](/assets/images/DSC07957.JPG)

## What this repo contains
- Scripts to run a COLMAP reconstruction from an image folder(images taken from the mip-Nerf360 dataset https://jonbarron.info/mipnerf360/), postprocess the dense mesh, and run a MuJoCo pick-and-place evaluation.

## Requirements
- Ubuntu (tested on 24.04)
- Python 3.12
- COLMAP installed and on PATH (https://github.com/colmap/colmap)
- MeshLab (for manual edits)
- Blender (for base flattening)
- MuJoCo

## Quick setup
1. Install COLMAP, GLOMAP, MiDaS, MuJoCo, Blender, MeshLab per their docs.
2. Put images in `assets`.
3. Run reconstruction:
    python midas_integrated_pipeline.py
4. Open `reconstruction.ply` with MeshLab and Blender for final edits (flatten base, fill holes, reorient mesh).
5. Place in sim_work/Simple-MuJoCo-PickNPlace/asset/panda/assets and run:
    python pnp.py
6. Remember to change file names accordingly within the python scripts and the object xml files.

## Quick Reproduction
1. clone this directory and simply run the pnp.py script after having installed Mujoco.

## Contact
For algorithm/code questions, mention this repo and provide sample images. Good luck!


