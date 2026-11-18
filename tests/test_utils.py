import math

import numpy as np
import pytest
import torch

from deepblur_napari._utils import (
    _disk_radii,
    cutoff_frequency,
    encircled_energy_radius,
    estimate_tiled_psfs,
    full_width_half_maximum,
    reflect_pad,
    tile_centers,
)
from deepblur_napari._widget import to_2d_image


def gaussian(sigma, center=(35.3, 34.6), size=71):
    yy, xx = torch.meshgrid(
        torch.arange(float(size)), torch.arange(float(size)), indexing="ij"
    )
    return torch.exp(
        -((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma**2)
    )


def coma_psf(waves, n=71, pad=4):
    """PSF of a circular pupil with a coma aberration of `waves` waves"""
    N = n * pad
    y, x = torch.meshgrid(
        torch.arange(N) - N // 2, torch.arange(N) - N // 2, indexing="ij"
    )
    rho = torch.sqrt(x**2 + y**2) / (N / (2 * pad))
    theta = torch.atan2(y, x)
    phase = waves * (3 * rho**3 - 2 * rho) * torch.cos(theta)
    pupil = (rho <= 1) * torch.exp(2j * math.pi * phase)
    psf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil)))
    psf = psf.abs()[N // 2 - n // 2 : N // 2 + n // 2 + 1] ** 2
    psf = psf[:, N // 2 - n // 2 : N // 2 + n // 2 + 1]
    return (psf / psf.sum()).float()


@pytest.mark.parametrize("sigma", [1.5, 3.0, 6.0])
def test_ee50_of_gaussians(sigma):
    radius, center = encircled_energy_radius(gaussian(sigma))
    # EE50 of a Gaussian integrated over the pixels
    exact = math.sqrt(sigma**2 + 1 / 12) * math.sqrt(2 * math.log(2))
    assert float(radius) == pytest.approx(exact, rel=0.02)
    assert center.tolist() == pytest.approx([35.3, 34.6], abs=0.3)


def test_ee50_is_l1_energy():
    psf = gaussian(3.0)
    signed = psf.clone()
    signed[0, 0] = -signed[0, 0]  # the energy uses |psf|
    assert float(encircled_energy_radius(signed)[0]) == pytest.approx(
        float(encircled_energy_radius(psf)[0]), abs=1e-3
    )


def test_ee50_optimizes_the_center_of_coma():
    psf = coma_psf(0.5)
    radius, center = encircled_energy_radius(psf)

    # Disk centered on the centroid, and brute force over centers
    s = 4
    yy, xx = torch.meshgrid(
        torch.arange(71.0), torch.arange(71.0), indexing="ij"
    )
    points = torch.stack((yy.flatten(), xx.flatten()), dim=1)
    offsets = (torch.arange(s) + 0.5) / s - 0.5
    sub = torch.stack(torch.meshgrid(offsets, offsets, indexing="ij"), -1)
    sub_points = (points[:, None] + sub.reshape(-1, 2)).reshape(-1, 2)
    energy = psf.flatten()
    sub_energy = (energy[:, None] / s**2).expand(-1, s**2).reshape(-1)
    centroid = (energy[:, None] * points).sum(0)
    at_centroid = _disk_radii(sub_points, sub_energy, centroid[None], 0.5)
    steps = torch.arange(-3, 3.01, 0.1)
    grid = torch.stack(torch.meshgrid(steps, steps, indexing="ij"), -1)
    brute = _disk_radii(
        sub_points, sub_energy, centroid + grid.reshape(-1, 2), 0.5
    ).min()

    assert float(radius) < 0.95 * float(at_centroid[0])
    assert float(radius) == pytest.approx(float(brute), abs=0.02)


def test_ee50_batch_shapes():
    psfs = torch.softmax(torch.randn(1, 3, 4, 51 * 51), -1)
    radii, centers = encircled_energy_radius(psfs.reshape(1, 3, 4, 51, 51))
    assert radii.shape == (1, 3, 4) and centers.shape == (1, 3, 4, 2)


@pytest.mark.parametrize(
    "sigma, rel", [(1.0, 0.03), (1.5, 0.015), (3.0, 0.01)]
)
@pytest.mark.parametrize("center", [(35.0, 35.0), (35.3, 34.6), (35.5, 35.5)])
def test_fwhm_of_gaussians(sigma, rel, center):
    # The peak between pixels must not widen the PSF (up to 12% with the
    # brightest pixel as maximum for sigma = 1)
    exact = 2 * math.sqrt(2 * math.log(2)) * sigma
    fwhm = float(full_width_half_maximum(gaussian(sigma, center)))
    assert fwhm == pytest.approx(exact, rel=rel)


def test_fwhm_of_elliptic_gaussian_and_airy():
    yy, xx = torch.meshgrid(
        torch.arange(71.0), torch.arange(71.0), indexing="ij"
    )
    u, v = (yy - 35.2 + xx - 34.7) / 2**0.5, (yy - 35.2 - xx + 34.7) / 2**0.5
    elliptic = torch.exp(-(u**2 / (2 * 2**2) + v**2 / (2 * 4**2)))
    geometric_mean = 2 * math.sqrt(2 * math.log(2)) * math.sqrt(2 * 4)
    assert float(full_width_half_maximum(elliptic)) == pytest.approx(
        geometric_mean, rel=0.01
    )
    # Airy disk: 1.029 lambda / D, with lambda / D = 4 pixels in coma_psf
    airy = full_width_half_maximum(coma_psf(0.0))
    assert float(airy) == pytest.approx(1.029 * 4, rel=0.01)


def test_fwhm_of_a_disk_and_shapes():
    yy, xx = torch.meshgrid(
        torch.arange(71.0), torch.arange(71.0), indexing="ij"
    )
    disk = ((yy - 35.2) ** 2 + (xx - 34.9) ** 2 <= 15**2).float()
    assert float(full_width_half_maximum(disk)) == pytest.approx(30, rel=0.02)
    fwhm = full_width_half_maximum(torch.rand(1, 3, 20, 51, 51))
    assert fwhm.shape == (1, 3, 20)


@pytest.mark.parametrize("sigma", [1.5, 3.0])
def test_cutoff_of_gaussians(sigma):
    # The MTF exp(-2 pi^2 sigma^2 f^2) reaches 5% at this frequency
    exact = math.sqrt(math.log(20) / 2) / (math.pi * sigma)
    cutoff = float(cutoff_frequency(gaussian(sigma)))
    assert cutoff == pytest.approx(exact, rel=0.01)


def test_cutoff_of_elliptic_and_diffraction_psfs():
    yy, xx = torch.meshgrid(
        torch.arange(71.0), torch.arange(71.0), indexing="ij"
    )
    elliptic = torch.exp(
        -((yy - 35) ** 2 / (2 * 1.5**2) + (xx - 35) ** 2 / (2 * 3**2))
    )
    geometric_mean = math.sqrt(math.log(20) / 2) / (math.pi * math.sqrt(4.5))
    assert float(cutoff_frequency(elliptic)) == pytest.approx(
        geometric_mean, rel=0.01
    )
    # Without aberrations, 5% of the MTF at 0.88 x 2 NA / lambda, and
    # NA / lambda = 1/8 cycles per pixel in coma_psf
    assert float(cutoff_frequency(coma_psf(0.0))) == pytest.approx(
        0.88 * 2 / 8, rel=0.02
    )
    # Undersampled: the MTF exceeds 5% up to the Nyquist frequency
    dirac = torch.zeros(71, 71)
    dirac[35, 35] = 1
    assert float(cutoff_frequency(dirac)) == 0.5
    assert cutoff_frequency(torch.rand(1, 3, 20, 51, 51)).shape == (1, 3, 20)


def test_reflect_pad_larger_than_image():
    y = torch.rand(1, 1, 10, 7)
    out = reflect_pad(y, 0, 300, 0, 310)
    assert out.shape == (1, 1, 310, 317)
    assert torch.equal(out[..., :10, :7], y)


def test_tiled_psfs_of_image_smaller_than_patch():
    def model(x):  # a PSF per patch
        return torch.softmax(torch.zeros(x.shape[0], 1, 51 * 51), -1).reshape(
            -1, 1, 51, 51
        )

    psfs, n_tiles = estimate_tiled_psfs(
        model, torch.rand(1, 3, 100, 120), (256, 256), (128, 128)
    )
    assert n_tiles == (1, 1) and psfs.shape == (1, 3, 1, 51, 51)


def test_tile_centers():
    centers = tile_centers((2, 3), (320, 320), (128, 64))
    assert centers.tolist() == [
        [159, 159],
        [159, 223],
        [159, 287],
        [287, 159],
        [287, 223],
        [287, 287],
    ]


def test_to_2d_image():
    assert to_2d_image(np.zeros((5, 6)), False).shape == (5, 6)
    assert to_2d_image(np.zeros((5, 6, 4)), True).shape == (5, 6, 3)
    assert to_2d_image(np.zeros((5, 6, 1)), False).shape == (5, 6)
    assert to_2d_image(np.zeros((10, 5, 6)), False) is None  # z-stack
    assert to_2d_image(np.zeros((2, 10, 5, 6)), False) is None
