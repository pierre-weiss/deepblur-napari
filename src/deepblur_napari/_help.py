"""Help of the DeepBlur widget, shown by its Help button (rich text of Qt)"""

HELP = """
<h2>DeepBlur</h2>
<p>Neural networks estimate the blur (point spread function, PSF) of an
image, which may vary across the field of view. The image is then deblurred
with this blur. The networks are downloaded on first use.</p>

<h3>Images</h3>
<ul>
<li>2D images only: grayscale (1 channel) or color (3 channels, RGB). The
alpha channel of RGBA images is dropped.</li>
<li>Other images (z-stacks, time series, images of 2 or more than 3
channels) are not processed, not even slice by slice: they stay in the
viewer, and DeepBlur leaves them aside.</li>
<li>Add images by drag and drop onto napari, or with File &gt; Open. Each
image joins the drop-down menu (000, 001...) and becomes the active image,
shown in the <i>image</i> layer.</li>
<li>The layers above it show the results of the active image only:
<i>deblur</i>, <i>psf</i> and the maps of metrology.</li>
</ul>

<h3>Identification model</h3>
<p>The network that estimates the PSF on patches of the image (256 or 320
pixels). Each one was trained on a family of blurs:</p>
<ul>
<li><b>z3_z36</b>: diffraction blurs, with optical aberrations described by
the Zernike polynomials 3 to 35. The best choice for microscopes and optical
blurs.</li>
<li><b>z3_z6</b> to <b>z3_z28</b>: fewer aberrations, from defocus and
astigmatism only (z3_z6) to higher orders.</li>
<li><b>chromatic_z3_z36</b>: another network for the aberrations of z3_z36,
trained for color channels with different blurs.</li>
<li><b>motion</b>: motion blurs (camera shake, moving sample).
<b>motion_x_z3_z16</b>: motion or diffraction blurs. Use one of them for
motion blurs.</li>
</ul>
<p><b>Model info</b> shows the training configuration of the model.</p>

<h3>Metrology</h3>
<p>Assesses the state of an optical system, e.g. a microscope, from the
image of a sample.</p>
<ul>
<li><b>Grid</b>: displays the PSFs estimated on a grid of points, spaced by
<i>PSF Spacing</i> pixels, in the <i>psf</i> layer.</li>
<li>Click on the image, with the <i>image</i> or <i>psf</i> layer selected,
to display the PSF at that point. <b>Delete All</b> removes the displayed
PSFs.</li>
<li><b>EE50 Map</b>: radius of the smallest disk containing 50% of the
energy of the PSF. Its center is optimized, so that it also fits asymmetric
PSFs (e.g. coma).</li>
<li><b>FWHM Map</b>: full width at half maximum. For PSFs of any shape, it
is the diameter of the disk whose area is that of the region where the PSF
exceeds half its maximum.</li>
<li><b>Cut-off Map</b>: highest frequency transmitted by the optics, in
cycles per pixel (at most 0.5, the Nyquist frequency): the radius of the
frequencies where the MTF, the modulus of the Fourier transform of the PSF,
exceeds 5%. Without aberrations, it is 0.88 &times; 2 NA / &lambda;, with
&lambda; in pixels. Aberrations lower it.</li>
</ul>
<p>The maps are computed on the tiles of the operator (below) and
interpolated between their centers. A message gives their range and
median.</p>

<h3>Operator estimation</h3>
<p>The image is cut into overlapping tiles, of the size of the patches of
the model and spaced by <i>psx</i> and <i>psy</i> pixels. The model
estimates the PSF of each tile. The blur operator is built from these
PSFs:</p>
<ul>
<li><b>constant</b>: a single PSF for the whole image, the median PSF of the
tiles (the one closest to all the others).</li>
<li><b>eigen_psf</b>: a principal component analysis of the PSFs of the
tiles gives a few eigen-PSFs. Their coefficients are interpolated over the
image, which gives a PSF at every pixel.</li>
<li><b>tiled_psf</b>: each tile is blurred with its own PSF, and the
overlapping tiles are blended smoothly. The PSF changes gradually from one
tile to the next.</li>
</ul>
<p><b>Estimate Operator</b> computes the operator of the active image and
shows its PSFs in the <i>psf</i> layer. Deblurring then uses it. Smaller
spacings follow faster changes of the blur, but take longer.</p>

<h3>Noise</h3>
<p>The noise model, in photon counts: y = Poisson(G x) + N(0, &sigma;<sup>2</sup>),
where x is the image scaled to [0, 1] and G = 2<sup>bit depth</sup> &minus; 1.</p>
<ul>
<li><b>Gaussian &sigma;</b>: standard deviation of the Gaussian (read-out)
noise, in photon counts.</li>
<li><b>Bit depth</b>: sets the Poisson noise, through the number G of photons
of the brightest pixel. The lower it is, the noisier the image. It is not the
bit depth of the image file.</li>
<li>The networks were trained in realistic regimes, where the Gaussian noise
is weak compared to the Poisson noise: &sigma; from 0.5 to 5, and bit depth
from 8 to 12 (up to 16 for some identification models). Outside these
ranges, the results are less reliable.</li>
<li><b>Estimate noise</b>: a neural network estimates &sigma; and the bit
depth directly from the active image.</li>
</ul>

<h3>Deblurring</h3>
<p><b>Deblur Image</b>, available once an operator is estimated, deblurs the
active image with this operator and the noise levels. The result is shown in
the <i>deblur</i> layer.</p>
<ul>
<li><b>Tikhonov</b>: quadratic regularization. Fast, but leaves noise and
ringing.</li>
<li><b>TV</b>: total variation regularization. Removes noise, but flattens
textures.</li>
<li><b>DPIR</b>: iterative algorithm with a pretrained denoising network
(DRUNet).</li>
<li><b>RAM</b>: a network trained for many inverse problems (Reconstruct
Anything Model), fine-tuned for DeepBlur. Usually the best results.</li>
</ul>
<p>Computations run in the background, with a progress bar at the bottom
right of napari. The buttons are disabled meanwhile.</p>
"""
