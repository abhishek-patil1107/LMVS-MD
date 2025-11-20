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



Applications/Software used: 
1. Meshlab: For editing vertices, filling in meshes, reorienting meshes.
2. Blender: Flatten vertices a the base to ensure stable grasping.
3. Spyder: Python code development IDE

repositories used:
1. XM-Code - https://github.com/ComputationalRobotics/XM-code
2. Colmap - https://github.com/colmap/colmap
3. Simple-MuJoCo-PickNPlace - https://github.com/volunt4s/Simple-MuJoCo-PickNPlace

Steps taken

1. Create python environment
2. Install requirements
3. Perform 3D reconstruction
4. Import reconstruction to Meshlab
5. Edit meshes in Meshlab
6. Flatten bottom surface in Blender
7. Import mesh into MuJoCo simulation for Pick and Place operation
