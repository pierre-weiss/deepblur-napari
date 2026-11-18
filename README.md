# DeepBlur for napari

[![License BSD-3](https://img.shields.io/pypi/l/deepblur-napari.svg?color=green)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI](https://img.shields.io/pypi/v/deepblur-napari.svg?color=green)](https://pypi.org/project/deepblur-napari)
[![Python Version](https://img.shields.io/pypi/pyversions/deepblur-napari.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/deepblur-napari)](https://napari-hub.org/plugins/deepblur-napari)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)

DeepBlur estimates the blur of an image with neural networks, without
calibration, and deblurs the image. The blur (point spread function, PSF)
may vary across the field of view, as in most microscopes. From the
estimated PSFs, the plugin also measures the quality of the optics: maps of
the EE50, of the FWHM and of the cut-off frequency.

## Usage

Open the plugin with Plugins > DeepBlur > Deep Blur, or with
`napari -w deepblur_napari`. It opens with demo images: add yours by drag
and drop. The **Help** button of the plugin explains every setting.

1. **Images**: DeepBlur processes 2D grayscale and RGB images. Each image
   added to napari joins the *Active Image* menu.
2. **Identification model**: the network that estimates the PSF. Use
   `z3_z36` for microscopes and optical blurs, `motion` or `motion_x_z3_z16`
   for motion blurs.
3. **Operator estimation**: *Estimate Operator* builds the blur operator of
   the image from the PSFs estimated on overlapping tiles. It uses a single
   PSF (`constant`), PSFs interpolated through eigen-PSFs (`eigen_psf`), or
   PSFs blended between the tiles (`tiled_psf`).
4. **Noise**: set the Gaussian noise and the bit depth, which sets the
   Poisson noise, or estimate them with *Estimate noise*.
5. **Deblurring**: *Deblur Image* with Tikhonov, TV, DPIR or RAM (usually
   the best).

**Metrology**: *Grid* and clicks on the image display the estimated PSFs.
*EE50 Map*, *FWHM Map* and *Cut-off Map* map the radius containing 50% of
the energy of the PSF, its full width at half maximum, and the cut-off
frequency of the optics.

Computations run in the background. The first use of a network downloads
its weights.

## Installation

DeepBlur needs Python 3.10 to 3.13 (PyTorch has no wheels for Intel Macs).
It runs on CPU and, much faster, on NVIDIA GPUs.

In an environment that has napari, install it from napari's plugin manager
(Plugins > Install/Uninstall Plugins), or with

```sh
pip install deepblur-napari
```

In a new environment, `pip install "deepblur-napari[all]"` also installs
napari with its Qt backend.

On Linux, the PyTorch wheels of PyPI use CUDA. To get a CPU-only PyTorch, or
one for another CUDA version, install it first by following
https://pytorch.org/get-started/locally/, then install the plugin.

### From the sources

[uv] creates the environment `.venv` with the versions locked in `uv.lock`:

```sh
git clone https://github.com/pierre-weiss/deepblur-napari.git
cd deepblur-napari
uv sync --extra all      # add --extra testing for the test tools
uv run napari -w deepblur_napari
```

`uv run python debug.py` opens the same window, e.g. to debug the plugin in
an editor. With pip, run `pip install -e ".[all]"` in a new environment, or
`pip install -e .` in an environment that has napari.

### Weights

The networks download their weights on first use from the MAMBO server
(https://mambo-utoulouse.fr/weights), or from the Hugging Face hub
(`mhnguyen712/deepblur`) if the server is unreachable. They are cached in
`~/.cache/deepblur/weights`. You can change the server and the cache folder
with the environment variables `DEEPBLUR_WEIGHTS_URL` and `DEEPBLUR_CACHE`.

## Contributing

Contributions are very welcome. Run the tests with `uv run pytest` (after
`uv sync --extra all --extra testing`) or with [tox], and please ensure the
coverage at least stays the same before you submit a pull request. The
tests download the weights of the networks the first time.

To release, push a tag: `git tag v0.1.0 && git push origin v0.1.0`. The
continuous integration runs the tests, builds the package from a clean
checkout (the version comes from the tag) and publishes it on PyPI.

## License

Distributed under the terms of the [BSD-3] license, deepblur-napari is free
and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed
description.

[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/pierre-weiss/deepblur-napari/issues
[tox]: https://tox.readthedocs.io/en/latest/
[uv]: https://docs.astral.sh/uv/
