r"""
Image conversions, tiled PSF estimation and PSF metrology.

The tiles are those of :func:`deepinv.utils.image_to_patches`, the tiling of
:class:`deepinv.physics.TiledSpaceVaryingBlur`, so that the PSF estimated on a
tile parametrizes the same tile of the operator.
"""

import numpy as np
import torch
import torch.nn.functional as F
from deepinv.utils import image_to_patches


def numpy_to_torch(image: np.ndarray) -> torch.Tensor:
    r"""``(H, W)`` or ``(H, W, C)`` array -> ``(1, C, H, W)`` float32 tensor."""
    if image.ndim == 2:
        image = image[None]
    elif image.ndim == 3 and image.shape[2] in (1, 2, 3, 4):
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return torch.from_numpy(image.astype(np.float32))[None]


def torch_to_numpy(tensor: torch.Tensor, idx: int = 0) -> np.ndarray:
    r"""``(B, C, H, W)`` tensor -> ``(H, W, C)`` array, or ``(H, W)`` if ``C = 1``."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {tensor.shape}")
    img = tensor[idx].permute(1, 2, 0).detach().cpu().numpy()
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    r"""Rescales each image of the batch to [0, 1] (all channels together)."""
    flat = x.reshape(x.size(0), -1)
    amin, amax = torch.aminmax(flat, dim=1, keepdim=True)
    denom = amax - amin
    flat = (flat - amin) / torch.where(denom < 1e-8, 1.0, denom)
    return flat.reshape(x.shape)


def to_ram_param(gain: float, sigma: float) -> tuple[float, float]:
    r"""
    Noise ``Poisson(G x) + N(0, sigma^2)`` in photon counts -> ``gain`` and
    ``sigma`` of :class:`deepinv.models.RAM` for images in [0, 1].
    """
    return 1 / gain, sigma / gain


def tile_centers(
    n_tiles: tuple[int, int],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    device=None,
) -> torch.Tensor:
    r"""
    Centers ``(y, x)`` of the tiles of :func:`deepinv.utils.image_to_patches`,
    in row-major order, as a ``(n_rows * n_cols, 2)`` tensor.
    """
    ys = (patch_size[0] - 1) // 2 + stride[0] * torch.arange(
        n_tiles[0], device=device
    )
    xs = (patch_size[1] - 1) // 2 + stride[1] * torch.arange(
        n_tiles[1], device=device
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((yy.flatten(), xx.flatten()), dim=1)


def reflect_pad(
    y: torch.Tensor, top: int, bottom: int, left: int, right: int
) -> torch.Tensor:
    r"""
    Pads ``y`` ``(B, C, H, W)`` by reflection, repeatedly if the padding is
    larger than the image (which :func:`torch.nn.functional.pad` refuses).
    """
    while top or bottom or left or right:
        H, W = y.shape[-2:]
        t, b = min(top, H - 1), min(bottom, H - 1)
        lt, rt = min(left, W - 1), min(right, W - 1)
        if not (t or b or lt or rt):  # 1-pixel image: nothing to reflect
            return F.pad(y, (left, right, top, bottom), mode="replicate")
        y = F.pad(y, (lt, rt, t, b), mode="reflect")
        top, bottom, left, right = top - t, bottom - b, left - lt, right - rt
    return y


def _normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    r"""Min-max normalizes each patch ``(..., h, w)`` to [0, 1], as in training."""
    p_min = patches.amin(dim=(-2, -1), keepdim=True)
    p_range = patches.amax(dim=(-2, -1), keepdim=True) - p_min
    return (patches - p_min) / torch.where(p_range > 0, p_range, 1.0)


@torch.no_grad()
def estimate_tiled_psfs(
    model: torch.nn.Module,
    y: torch.Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    batch_size: int = 8,
) -> tuple[torch.Tensor, tuple[int, int]]:
    r"""
    Estimates one PSF per tile and per channel, like
    ``deepinv.models.DeepBlur.forward_tiled`` (deepinv PR #1494) but for any
    identification network: each channel of each tile is min-max normalized
    and processed as a grayscale image.

    An image smaller than a patch is reflected at the bottom and right to
    the patch size, and has a single tile.

    :param torch.Tensor y: blurry image ``(B, C, H, W)``.
    :return: the PSFs ``(B, C, K, psf_h, psf_w)`` of the ``K = n_rows * n_cols``
        tiles in row-major order, and ``(n_rows, n_cols)``.
    """
    H, W = y.shape[-2:]
    y = reflect_pad(
        y, 0, max(0, patch_size[0] - H), 0, max(0, patch_size[1] - W)
    )
    patches = image_to_patches(y, patch_size, stride)
    B, C, n_rows, n_cols = patches.shape[:4]
    patches = _normalize_patches(patches.reshape(-1, 1, *patch_size))
    psfs = torch.cat([model(p) for p in patches.split(batch_size)])
    return psfs.reshape(B, C, n_rows * n_cols, *psfs.shape[-2:]), (
        n_rows,
        n_cols,
    )


@torch.no_grad()
def estimate_psfs_at_points(
    model: torch.nn.Module,
    y: torch.Tensor,
    centers: torch.Tensor,
    patch_size: tuple[int, int],
    batch_size: int = 8,
) -> torch.Tensor:
    r"""
    Estimates the PSF of each channel on the patches centered on ``centers``
    (``(N, 2)``, as ``(y, x)``), and returns an image ``(B, C, H, W)`` showing
    each PSF at its center (maximum where PSFs overlap). The image is
    reflected at the borders. The patches go through the model by batches of
    ``batch_size`` points.
    """
    B, C, H, W = y.shape
    ph, pw = patch_size
    y_pad = reflect_pad(y, ph // 2, ph // 2, pw // 2, pw // 2)
    out = torch.zeros_like(y)
    points = centers.tolist()
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        patches = torch.stack(
            [y_pad[..., i : i + ph, j : j + pw] for i, j in batch], dim=2
        )  # (B, C, n, ph, pw)
        psfs = model(_normalize_patches(patches.reshape(-1, 1, ph, pw)))
        h, w = psfs.shape[-2:]
        psfs = psfs.reshape(B, C, len(batch), h, w)

        for k, (i, j) in enumerate(batch):
            # PSF pixels inside the image
            i0, j0 = i - h // 2, j - w // 2
            top, left = max(0, -i0), max(0, -j0)
            bottom, right = min(h, H - i0), min(w, W - j0)
            region = out[..., i0 + top : i0 + bottom, j0 + left : j0 + right]
            region.copy_(
                torch.maximum(region, psfs[:, :, k, top:bottom, left:right])
            )
    return out


def encircled_energy_radius(
    psfs: torch.Tensor,
    fraction: float = 0.5,
    subsampling: int = 4,
    tol: float = 1 / 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Radius, in pixels, of the smallest disk containing ``fraction`` of the
    energy of each PSF (EE50 for ``fraction = 0.5``), and the center of this
    disk.

    The energy is the L1 norm: the sum of the absolute values of the pixels.
    The center of the disk is optimized, since for asymmetric PSFs (e.g. coma)
    it is neither the centroid nor the peak: a search on a 1-pixel grid
    covering the centroid and the peak is refined on 5 x 5 grids of halving
    steps, down to ``tol`` pixels.

    The energy of each pixel is spread uniformly over ``subsampling**2``
    sub-pixels, which matters for PSFs of a few pixels, and the pixels outside
    the brightest ones holding 99.9% of the energy are ignored.

    :param torch.Tensor psfs: PSFs ``(..., h, w)``.
    :return: (:class:`torch.Tensor`) radii ``(...)`` and centers ``(..., 2)``,
        as ``(y, x)`` in pixels.
    """
    shape = psfs.shape[:-2]
    disks = [
        _smallest_disk(psf, fraction, subsampling, tol)
        for psf in psfs.reshape(-1, *psfs.shape[-2:])
    ]
    radii = torch.stack([radius for radius, _ in disks]).reshape(shape)
    centers = torch.stack([center for _, center in disks])
    return radii, centers.reshape(*shape, 2)


def _smallest_disk(
    psf: torch.Tensor, fraction: float, subsampling: int, tol: float
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Radius and center of the smallest disk containing ``fraction`` of the energy of ``psf`` ``(h, w)``."""
    kw = {"device": psf.device, "dtype": psf.dtype}
    w = psf.shape[-1]
    energy = psf.abs().flatten()
    energy, idx = (energy / energy.sum().clamp(min=1e-12)).sort(
        descending=True
    )
    n = int((energy.cumsum(0) < 0.999).sum()) + 1
    energy, idx = energy[:n], idx[:n]
    points = torch.stack((idx // w, idx % w), dim=1).to(psf.dtype)

    # Coarse search on the pixels, over the box containing the centroid and
    # the peak (with a margin of 2 pixels)
    centroid = (energy[:, None] * points).sum(0) / energy.sum()
    peak = points[0]
    lo = torch.minimum(centroid, peak) - 2
    hi = torch.maximum(centroid, peak) + 2
    step = max(1.0, float((hi - lo).max()) / 8)
    ys = torch.arange(float(lo[0]), float(hi[0]) + step / 2, step, **kw)
    xs = torch.arange(float(lo[1]), float(hi[1]) + step / 2, step, **kw)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    candidates = torch.cat([grid.reshape(-1, 2), centroid[None], peak[None]])
    radii = _disk_radii(points, energy, candidates, fraction)
    center = candidates[radii.argmin()]

    # Refinement on the sub-pixels, around the best center so far
    s = subsampling
    offsets = (torch.arange(s, **kw) + 0.5) / s - 0.5
    sub = torch.stack(torch.meshgrid(offsets, offsets, indexing="ij"), dim=-1)
    sub_points = (points[:, None] + sub.reshape(-1, 2)).reshape(-1, 2)
    sub_energy = (energy[:, None] / s**2).expand(-1, s**2).reshape(-1)
    moves = torch.arange(-2, 3, **kw)
    moves = torch.stack(torch.meshgrid(moves, moves, indexing="ij"), dim=-1)
    moves = moves.reshape(-1, 2)
    while True:
        step /= 2
        candidates = center + step * moves
        radii = _disk_radii(sub_points, sub_energy, candidates, fraction)
        center = candidates[radii.argmin()]
        if step <= tol:
            return radii.min(), center


def _disk_radii(
    points: torch.Tensor,
    energy: torch.Tensor,
    centers: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    r"""
    Radii ``(M,)`` of the disks centered on ``centers`` ``(M, 2)`` that contain
    ``fraction`` of the energy, for point masses ``energy`` ``(N,)`` at
    ``points`` ``(N, 2)`` with a total energy of (nearly) 1. The radius is
    linearly interpolated between the two points where the encircled energy
    crosses ``fraction``.
    """
    dist = ((centers[:, None] - points[None]) ** 2).sum(-1).sqrt()  # (M, N)
    dist, order = dist.sort(dim=-1)
    encircled = energy[order].cumsum(-1)

    # First point where the encircled energy reaches the fraction, and the
    # previous one (radius and energy 0 before the first one)
    target = torch.full_like(encircled[:, :1], fraction)
    i1 = torch.searchsorted(encircled, target).clamp(
        max=encircled.size(-1) - 1
    )
    i0 = (i1 - 1).clamp(min=0)
    first = i1 == 0
    e1, r1 = encircled.gather(-1, i1), dist.gather(-1, i1)
    e0 = torch.where(first, 0.0, encircled.gather(-1, i0))
    r0 = torch.where(first, 0.0, dist.gather(-1, i0))
    t = ((target - e0) / (e1 - e0).clamp(min=1e-12)).clamp(0, 1)
    return (r0 + t * (r1 - r0)).squeeze(-1)


def full_width_half_maximum(
    psfs: torch.Tensor, upsampling: int = 8, chunk: int = 16
) -> torch.Tensor:
    r"""
    Full width at half maximum, in pixels, of PSFs of any shape: the diameter
    of the disk whose area is that of the region where the PSF is at least
    half its maximum, :math:`2 \sqrt{A / \pi}`.

    It is the usual FWHM for circular PSFs (Gaussian, Airy disk), and the
    geometric mean of the FWHMs along the axes for elliptic Gaussians. The
    region gathers all the parts above half the maximum: several lobes for
    a motion blur, a ring for a strong defocus.

    The maximum and the area are measured on the Fourier (sinc) interpolation
    of the PSF on a grid ``upsampling`` times finer: for PSFs of a few
    pixels, the peak is often between two pixels, above the brightest one.
    This interpolation is exact for band-limited PSFs, like diffraction PSFs
    sampled at the Nyquist rate. For PSFs with sharp edges (uniform disk or
    segment), it overshoots near the edges and gives a FWHM a few percent
    too small.

    :param torch.Tensor psfs: PSFs ``(..., h, w)``.
    :return: (:class:`torch.Tensor`) FWHMs ``(...)``.
    """
    shape = psfs.shape[:-2]
    h, w = psfs.shape[-2:]
    H, W = upsampling * h, upsampling * w
    top, left = H // 2 - h // 2, W // 2 - w // 2
    areas = []
    for batch in psfs.reshape(-1, h, w).split(chunk):
        # Zero padding of the centered spectrum
        spectrum = torch.fft.fftshift(torch.fft.fft2(batch), dim=(-2, -1))
        padded = spectrum.new_zeros(len(batch), H, W)
        padded[:, top : top + h, left : left + w] = spectrum
        fine = torch.fft.ifft2(torch.fft.ifftshift(padded, dim=(-2, -1))).real
        half = fine.amax(dim=(-2, -1), keepdim=True) / 2
        areas.append((fine >= half).sum(dim=(-2, -1)) / upsampling**2)
    areas = torch.cat(areas).to(psfs.dtype).reshape(shape)
    return 2 * (areas / torch.pi).sqrt()


def cutoff_frequency(
    psfs: torch.Tensor,
    threshold: float = 0.05,
    padding: int = 4,
    chunk: int = 16,
) -> torch.Tensor:
    r"""
    Cut-off frequency of PSFs, in cycles per pixel: the radius of the disk
    whose area is that of the frequencies where the modulation transfer
    function (MTF, modulus of the Fourier transform of the PSF, 1 at the
    zero frequency) is at least ``threshold``. It is at most 0.5, the Nyquist
    frequency, reached by undersampled PSFs.

    These frequencies may form an ellipse (astigmatism, motion) or rings
    (defocus, whose MTF vanishes and rises again): they all count. The PSFs
    are zero-padded to ``padding`` times their size, to sample the MTF
    finely.

    The MTF of an aberration-free system vanishes at :math:`2 \mathrm{NA} /
    \lambda` (:math:`\lambda` in pixels), and reaches 5% at 0.88 times this
    frequency. Aberrations lower the cut-off. Lower thresholds come closer to
    :math:`2 \mathrm{NA} / \lambda`, but the MTFs of the PSFs estimated by
    the networks are not accurate enough there: on blurred and noisy
    synthetic images, 5% gave the most accurate cut-offs (median error -3%).

    :param torch.Tensor psfs: PSFs ``(..., h, w)``.
    :return: (:class:`torch.Tensor`) cut-off frequencies ``(...)``.
    """
    shape = psfs.shape[:-2]
    h, w = psfs.shape[-2:]
    areas = []
    for batch in psfs.reshape(-1, h, w).split(chunk):
        otf = torch.fft.fft2(batch, s=(padding * h, padding * w))
        mtf = otf.abs() / otf[:, :1, :1].abs().clamp(min=1e-12)
        areas.append((mtf >= threshold).float().mean(dim=(-2, -1)))
    areas = torch.cat(areas).to(psfs.dtype).reshape(shape)
    return (areas / torch.pi).sqrt().clamp(max=0.5)


def interpolate_tile_map(
    values: torch.Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    img_size: tuple[int, int],
) -> torch.Tensor:
    r"""
    Bilinear interpolation over the image of values given at the tile centers
    (see :func:`tile_centers`), constant beyond the outer centers.

    :param torch.Tensor values: values ``(B, C, n_rows, n_cols)``.
    :return: (:class:`torch.Tensor`) map ``(B, C, H, W)``.
    """
    B, C, n_rows, n_cols = values.shape

    def grid_coords(size, n, patch, step):
        # Pixel positions in tile units from the first center, in [-1, 1]
        pos = (
            torch.arange(size, device=values.device) - (patch - 1) // 2
        ) / step
        if n == 1:
            return torch.zeros_like(pos)
        return (2 * pos / (n - 1) - 1).clamp(-1, 1)

    gy = grid_coords(img_size[0], n_rows, patch_size[0], stride[0])
    gx = grid_coords(img_size[1], n_cols, patch_size[1], stride[1])
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).expand(B, -1, -1, -1)
    return F.grid_sample(
        values, grid.to(values.dtype), mode="bilinear", align_corners=True
    )
