# The Eyes Have It
An intuitive approach for 3D Occupancy Detection
## Snapshot
### Motivation
**1.** Generate BEV freespace via 2D Segmentation is commonly used in adas industry, but it has limitations in many real-world scenarios.

**2.** 3D Occupancy has been proven to be a superior alternative to the previous perception scheme.
### Goal
**1.** A 3D OCC approach that balances accuracy, inference speed, deployability and simplicity.

**2.** A baseline that could be trained on generalized GPUs.

**3.** CVPR 2023 OCC challenge!

**4.** Deployed on an automotive-grade platform with real-time fps.
### Method
**1.** Inverse MatrixVT with complexity：
$$(N_c * H * W) * (Z * X * Y) * C$$
or doing VT as below and following a depthnet
$$(N_c * H * W) * (Z * \Theta) * C$$
**2.** Sparse supervision on lidar annotations
### Experiments
TBD
