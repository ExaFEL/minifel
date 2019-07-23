# ExaFEL proxy app repository

## Description
The ExaFEL proxy app, MiniFEL, is a lightweight Python application that illustrates the process of real-time reconstructions from X-ray Free Electron Laser (XFEL) diffraction patterns at the Linac Coherent Light Source (LCLS), at SLAC.
It performs a small subset of the tasks in the MTIP algorithm for reconstructing a 3D electron density from 2D diffraction patterns of randomly oriented single particles. 

The app reads in thousands of simulated diffraction images of a single particle with Legion-based Psana2 framework in xtc2 format. 
It then merges these 2D diffraction images into a 3D diffraction volume.
The unknown phase information is recovered by an algorithm called Hybrid-Input-Output (HIO), Error Reduction(ER), or a combination of both, leading to the reconstruction of the 3D electron density of the particle.

The core of these algorithms involve many iterations of a simple loop, which combines FFT, IFFT, and simple arithmetic operations on 3D complex data, all available in numpy.
A GPU implementation of the loop is also possible by using cupy as an almost drop-in replacement for numpy, based on CUDA.

MiniFEL leverages the parallel programming abilities of Legion, a parallel programming system with a focus on data, portability, and high performance on heterogeneous architectures.

Although currently working on simulated data, the app aims at loading diffraction data as it arrives from the beam line and process it in real time.
Moving forward, we will gradually increase its complexity in order to be able to provide LCLS-II users with rapid feedback to let them take the best out of their limited time on the beamline.

## Build and run instructions
```
git submodule update --init
./setup/build_from_scratch.sh
./scripts/clean_build.sh
./scripts/run_cpu.sh
```
