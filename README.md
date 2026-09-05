# Tomo Reconstruction Toolkit

This repository contains a collection of research-oriented Python scripts for X-ray micro-tomography reconstruction, dark/bright field correction, slice-by-slice reconstruction, and post-processing for CHESS/near-field tomography workflows.

It is not a polished package install; rather, it is a working set of analysis scripts and helper utilities intended to be adapted to a specific beamline dataset or experimental setup.

## What this project does

The workflow in this repository is centered on:

- reading projection data from a tomography scan
- creating dark-field and bright-field corrections
- converting raw radiographs into a usable sinogram stack
- selecting a region of interest (ROI)
- reconstructing 2D slices with filtered back projection (FBP)
- tuning the rotation center and image-processing parameters
- optionally denoising, thresholding, stitching, and exporting reconstructed volumes

The code is designed around tomography data acquisition patterns from CHESS-style near-field scans, with helper functions in `tomoFunctions2.py` and operational scripts for specific reconstruction tasks.

## Repository layout

- `tomoFunctions2.py` — core tomography utilities, reconstruction helpers, and parameter-tuning GUI logic
- `first_quicktomo.py` — a compact, single-script reconstruction workflow for a first-pass tomographic reconstruction
- `serial_quicktomo.py` — a more extended serial reconstruction workflow for per-layer processing and center tuning
- `create_and_stitch_vtk_cli.py` — volume creation/export workflow for later visualization or stitching
- `create_and_stitch_vtk_cli_sept.py` — a second variant of the reconstruction/stitching workflow
- `preprocess_tomo_mask.py` — mask-based reconstruction and near-field tomography processing logic
- `denoise_cli.py` — denoising, thresholding, and binary volume preparation for downstream analysis
- `Make_reduced_copies.py` — utility for making reduced-resolution copies of data for faster iterations

## Typical reconstruction workflow

A typical processing path in this project looks like this:

1. Define data folders and scan metadata
   - dark-field image range
   - bright-field image range
   - tomographic projections
   - detector geometry and pixel size
   - start/end angle and number of projections

2. Generate dark and bright field images
   - median or averaged correction frames are created from the reference images

3. Build the radiograph stack
   - transmission data is normalized with the dark/bright corrections
   - a stack of attenuation radiographs is assembled

4. Select the region of interest
   - choose the detector window containing the sample and relevant features
   - often the ROI is used to reduce memory use and speed up reconstruction

5. Reconstruct one representative slice
   - estimate the rotation center
   - tune the sinogram cutoff and filtering parameters
   - inspect the reconstruction visually and refine the center

6. Reconstruct the full volume
   - iterate across slices using the chosen center and calibration values
   - save intermediate or final reconstructions to `.npy` files

7. Post-process
   - ring removal
   - noise suppression
   - thresholding / segmentation
   - volume stitching or VTK export for visualization

## What is most useful in this repo

The strongest reusable pieces are:

- `tomoFunctions2.py` for spatial ROI handling and reconstruction helpers
- the per-layer reconstruction logic used in `serial_quicktomo.py`
- the parameter-tuning approach for center estimation and image filtering
- the export/visualization pipeline for 3D arrays

## Dependencies

This project depends on scientific Python packages, including:

- `numpy`
- `scipy`
- `matplotlib`
- `tomopy`
- `scikit-image`
- `h5py`
- `PyEVTK` for VTK export in some workflows

The scripts are historically written for older Python environments, and several files use Python 2 conventions (`print x` syntax, old-style imports, and hard-coded paths). Some of the files are experimental and may require compatibility fixes in a modern Python environment.

## Setup

1. Create a Python environment with the scientific stack installed.
2. Clone or copy this directory to your analysis machine.
3. Adjust the dataset paths inside the script you are using.
4. Verify the detector parameters:
   - `nrows`
   - `ncols`
   - `pixel_size`
   - `start_tomo_ang`
   - `end_tomo_ang`
   - `tomo_num_imgs`
5. Run the script from the project directory.

## How to run the main scripts

### `first_quicktomo.py`

This is the most straightforward starting point for a first pass reconstruction. It contains a compact workflow for:

- dark/bright correction
- radiograph generation
- ROI cropping
- single-slice reconstruction
- center tuning and visual inspection

Typical usage:

```bash
python first_quicktomo.py
```

Then edit the hard-coded file paths and geometry values at the top of the script before executing.

### `serial_quicktomo.py`

This script is more exhaustive and aimed at multi-layer reconstruction. It is useful when a dataset needs per-layer processing, manual center sweeps, and array saving for later analysis.

Typical usage:

```bash
python serial_quicktomo.py
```

### `create_and_stitch_vtk_cli.py`

This workflow is intended for generating visualization-ready volumes or stitched outputs after reconstruction.

Typical usage:

```bash
python create_and_stitch_vtk_cli.py
```

## Recommended workflow for a new dataset

1. Start with `first_quicktomo.py` and use a single representative layer.
2. Tune the reconstruction center and threshold settings visually.
3. Once the slice looks correct, move to `serial_quicktomo.py` for full-volume reconstruction.
4. Save intermediate arrays (`.npy`) as checkpoints.
5. Use `denoise_cli.py` or a custom post-processing routine for signal cleanup and segmentation.
6. Export final volumes for 3D visualization or analysis.

## Notes and caveats

- This project relies heavily on hard-coded paths and experiment-specific values.
- Several scripts are not command-line tools in the usual sense; they are effectively analysis notebooks embedded as runnable Python files.
- Some scripts are Python 2-era code and may require modernization if run in a modern environment.
- The actual reconstruction quality depends strongly on detector calibration, rotation center estimation, and image normalization.

## Suggested next steps

If you want to turn this into a more maintainable project, the next useful improvements would be:

- add a small configuration file for dataset parameters
- convert the scripts into a single reusable reconstruction pipeline
- add CLI arguments instead of hard-coded values
- standardize output naming and directory structure
- add documentation for each script and expected input/output format

## License

No explicit license file is present in this directory, so use this code under the project conventions of your host institution or lab unless otherwise specified.

## Summary

This repository is best understood as a tomography-analysis toolbox for CHESS-style near-field experiments: it is built to reconstruct, inspect, and process X-ray projection data in a research setting, with emphasis on flexibility and experimental tuning rather than polished packaging.
