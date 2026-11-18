"""
This is the main function generating and handling the deepblur plugin
"""

import os
import imageio.v3 as io
import yaml
import torch
import numpy as np
import contextlib

# Napari dependencies
import napari
from magicgui.widgets import (
    PushButton,
    Container,
    ComboBox,
    Label,
    TextEdit,
    SpinBox,
    FloatSpinBox,
)
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

# DeepInv dependencies
import deepinv as dinv
from deepinv.physics.functional.interp import ThinPlateSpline
from deepinv.physics.blur import SpaceVaryingBlur, TiledSpaceVaryingBlur

from ._help import HELP

# DeepBlur networks and tools, shipped with the plugin until they are in deepinv
from ._models import ID_MODELS, load_id_model, NoiseLevelIdNet
from ._weights import get_weights, WeightsUnavailableError
from ._utils import (
    encircled_energy_radius,
    estimate_psfs_at_points,
    cutoff_frequency,
    estimate_tiled_psfs,
    full_width_half_maximum,
    interpolate_tile_map,
    min_max_normalize,
    numpy_to_torch,
    tile_centers,
    to_ram_param,
    torch_to_numpy,
)

# RAM was fine-tuned separately for grayscale and colour images
RAM_CHECKPOINTS = {1: "RAM/checkpoint.pth", 3: "RAM/checkpoint_rgb.pth"}
NOISE_ID_CHECKPOINT = "noise_identification/checkpoint.pth"

# Images loaded at startup, in this order, relative to the images/ folder,
# with the settings that work best for each (identification model, operator)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
DEFAULT_IMAGES = [
    "grayscale/microscopy/cd_blurry.png",  # z3_z36, tiled_psf: a hidden city
    "grayscale/microscopy/cm_blurry.png",  # motion, tiled_psf
    "grayscale/microscopy/svm_blurry.png",  # motion, tiled_psf
    "grayscale/microscopy/xp_04_0.png",  # z3_z36, tiled_psf
    "grayscale/microscopy/xp_13_0.png",  # z3_z36, tiled_psf
    "color/Blurry2_2.png",  # z3_z36, tiled_psf
    "color/Blurry1_2.png",  # z3_z36, tiled_psf
    "color/Blurry4_3.png",  # z3_z36, tiled_psf
]

# Results of an image shown in their own layer, from bottom to top above the
# image, with the settings of these layers. A layer only exists while the
# active image has its result.
RESULT_LAYERS = {
    "deblur": {"name": "deblur", "opacity": 0.2, "blending": "opaque"},
    "psf": {"name": "psf", "opacity": 0.2, "colormap": "turbo", "gamma": 0.5},
    "ee50": {
        "name": "EE50 radius (px)",
        "opacity": 0.5,
        "colormap": "viridis",
    },
    "fwhm": {"name": "FWHM (px)", "opacity": 0.5, "colormap": "viridis"},
    "cutoff": {
        "name": "Cut-off (cycles/px)",
        "opacity": 0.5,
        "colormap": "viridis",
    },
}

######################################################################################
#############                        DEEP-BLUR WIDGETS                ################
######################################################################################


class DeepBlurWidget(Container):
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer

        # Model and device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float32
        self._loading = False

        self.default_identification_model = "z3_z36"
        self.default_estimator_model = "constant"
        self.default_deblur_model = "RAM"

        self.data_normalization_fn = min_max_normalize

        self.list_deepblur = []

        # Deblurring and noise identification networks are loaded on first
        # use and then reused
        self._deblur_models = {}
        self._noise_id_model = None

        # The identification model is loaded in the background once the
        # widget is built (see _load_model)
        self.identification_model = None
        self.identification_model_properties = None
        self._model_name = None
        self._busy = False  # a computation runs (see _run)

        # Create widgets
        self.identification_model_selection = ComboBox(
            label="",
            choices=ID_MODELS,
            value=self.default_identification_model,
        )

        self.interpolation_model_selection = ComboBox(
            label="",
            choices=["constant", "tiled_psf", "eigen_psf"],
            value=self.default_estimator_model,
        )

        self.deblur_model_selection = ComboBox(
            label="Model",
            choices=["Tikhonov", "TV", "DPIR", "RAM"],
            value=self.default_deblur_model,
        )

        self.psf_spacing = SpinBox(
            label="PSF Spacing",
            value=256,
            min=64,
            max=1024,
            step=16,
            tooltip="Spacing of the PSFs displayed by Grid, in pixels",
        )

        self.operator_spacing_y = SpinBox(
            label="psy",
            value=128,
            min=64,
            max=1024,
            step=16,
        )

        self.operator_spacing_x = SpinBox(
            label="psx",
            value=128,
            min=64,
            max=1024,
            step=16,
        )

        # Noise parameters, as in the gradio app (see _deblur_action)
        self.gaussian_noise_level = FloatSpinBox(
            label="Gaussian σ",
            value=0.05,
            min=0.0,
            max=2**15 - 1,  # follows the bit depth, see _on_bit_depth_changed
            step=0.1,
            tooltip="Std of the Gaussian noise, in photon counts: up to "
            "2**bit_depth - 1",
        )
        # 3 decimals for the estimated values (the step alone gives 2)
        self.gaussian_noise_level.native.setDecimals(3)

        self.bit_depth = SpinBox(
            label="Bit depth",
            value=15,
            min=2,
            max=16,
            step=1,
            tooltip="Poisson noise: the image spans 2**bit_depth - 1 photon "
            "counts",
        )

        self.help_button = PushButton(label="Help")
        self._help_dialog = None  # created on first use, see _show_help
        self.display_model_info = PushButton(label="Model info")
        self.display_psf_grid = PushButton(label="Grid")
        self.estimate_operator = PushButton(label="Estimate Operator")
        self.delete_all_button = PushButton(label="Delete All")
        self.save_results = PushButton(label="Save Results")
        self.deblur_action = PushButton(label="Deblur Image")
        self.identify_noise = PushButton(
            label="Estimate noise",
            tooltip="Sets the Gaussian σ and the bit depth (Poisson gain) "
            "estimated on the active image",
        )
        self.ee50_map = PushButton(
            label="EE50 Map",
            tooltip="Map of the radius containing 50% of the PSF energy, "
            "interpolated between the tiles (psx, psy)",
        )
        self.fwhm_map = PushButton(
            label="FWHM Map",
            tooltip="Map of the full width at half maximum of the PSF: "
            "diameter of the disk with the area where the PSF exceeds half "
            "its maximum, interpolated between the tiles (psx, psy)",
        )
        self.cutoff_map = PushButton(
            label="Cut-off Map",
            tooltip="Map of the cut-off frequency, in cycles per pixel: "
            "radius of the frequencies where the MTF exceeds 5%, "
            "interpolated between the tiles (psx, psy)",
        )

        # and their callbacks
        self.help_button.clicked.connect(self._show_help)
        self.display_model_info.changed.connect(self._display_info)
        self.display_psf_grid.changed.connect(self._display_psf_grid)
        self.estimate_operator.changed.connect(self._estimate_operator)
        self.identification_model_selection.changed.connect(self._load_model)
        self.delete_all_button.clicked.connect(self._delete_all_callback)
        self.save_results.clicked.connect(self._save_results_callback)
        self.deblur_action.clicked.connect(self._deblur_action)
        self.identify_noise.clicked.connect(self._identify_noise)
        self.bit_depth.changed.connect(self._on_bit_depth_changed)
        self._on_bit_depth_changed()
        self.ee50_map.clicked.connect(self._display_ee50_map)
        self.fwhm_map.clicked.connect(self._display_fwhm_map)
        self.cutoff_map.clicked.connect(self._display_cutoff_map)

        super().__init__(layout="vertical")

        self._initialize_deepblur_layers()  # loads default images and creates the layers
        self._build_ui()
        self._update_widget()
        self._display_current()
        QTimer.singleShot(0, self._update_widget)
        self.viewer.layers.events.inserted.connect(self._on_image_load)

        # Disabled while a computation runs (see _set_busy)
        self._actions = [
            self.image_selector,
            self.identification_model_selection,
            self.display_psf_grid,
            self.delete_all_button,
            self.ee50_map,
            self.fwhm_map,
            self.cutoff_map,
            self.estimate_operator,
            self.identify_noise,
        ]
        self._load_model()

    def _initialize_deepblur_layers(self):
        self.image_layer = None
        self.result_layers = dict.fromkeys(RESULT_LAYERS)  # see _show_result
        self._displayed_obj = None  # DeepBlurObject shown in the layers

        # The image layers already open become deepblur images
        existing_images = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        for layer in existing_images:
            if self._add_image(_layer_image(layer), layer.name):
                self.viewer.layers.remove(layer)

        if not self.list_deepblur:
            for path in DEFAULT_IMAGES:
                image = io.imread(os.path.join(IMAGES_DIR, path))
                rgb = image.ndim == 3 and image.shape[-1] in (3, 4)
                self._add_image(to_2d_image(image, rgb), path)

    def _on_image_load(self, event):
        layer = event.value
        # Only the images added by the user: their other layers (points,
        # labels...) and the layers of this widget stay as they are
        if self._loading or not isinstance(layer, napari.layers.Image):
            return

        image = _layer_image(layer)
        if image is None:
            self._add_image(image, layer.name)  # explains why it is ignored
            return

        self._loading = True
        self.image_selector.changed.disconnect(self._on_image_selector_changed)
        self.viewer.layers.events.inserted.disconnect(self._on_image_load)
        QTimer.singleShot(
            0, lambda: self._remove_layer_and_update(layer, image)
        )

    def _remove_layer_and_update(self, layer, image):
        self.viewer.layers.remove(layer)
        self._add_image(image, layer.name)
        self._update_widget()
        self._display_current()
        # Callbacks back on
        self.image_selector.changed.connect(self._on_image_selector_changed)
        self.viewer.layers.events.inserted.connect(self._on_image_load)
        self._loading = False

    def _add_image(self, image, name):
        """
        Adds image, a 2D grayscale or RGB array (see to_2d_image), as the
        active deepblur image. Returns False, with a message, if image is
        None (not a 2D image).
        """
        if image is None:
            show_info(
                f"'{name}' is not a 2D grayscale or RGB image: DeepBlur "
                "leaves it aside"
            )
            return False
        self.list_deepblur.append(DeepBlurObject(image))
        self.idx_active = len(self.list_deepblur) - 1
        return True

    def _add_layer(self, data, **kwargs):
        """Adds an image layer without _on_image_load taking it as a new image"""
        loading, self._loading = self._loading, True
        try:
            return self.viewer.add_image(data, **kwargs)
        finally:
            self._loading = loading

    def _display_current(self):
        # Keep the contrast of the image we leave, to restore it when coming back
        if self._displayed_obj is not None:
            self._save_display(self._displayed_obj)

        obj = self.gco()
        self._displayed_obj = obj
        self._display_image(obj, obj.display.get("image"))

        # DEBLUR, PSF and the maps: only the results of this image
        for key in RESULT_LAYERS:
            self._show_result(key, obj.display.get(key))

    def _display_image(self, obj, display=None):
        """Shows obj.image in the image layer, created again if deleted"""
        if self.image_layer in self.viewer.layers:
            self.safe_update_layer(self.image_layer, obj.image, display)
        else:
            is_rgb = obj.image.ndim == 3
            self.image_layer = self._add_layer(
                obj.image,
                name="image",
                rgb=is_rgb,
                colormap=None if is_rgb else "gray",
            )
            # Below the results
            self.viewer.layers.move(
                self.viewer.layers.index(self.image_layer), 0
            )
            if display is not None:
                self.safe_update_layer(self.image_layer, obj.image, display)
        # Clicking on the image estimates the PSF there (see _show_result
        # for the psf layer)
        if self._on_mouse_press not in self.image_layer.mouse_drag_callbacks:
            self.image_layer.mouse_drag_callbacks.append(self._on_mouse_press)

    def _show_result(self, key, display=None):
        """
        Shows the result `key` of the active image (see RESULT_LAYERS) in
        its layer, creating the layer if needed, or
        removes the layer if the image has no such result.

        Parameters
        ----------
        key : str
        display : dict, optional
            Contrast settings saved by _save_display. If None, as for a new
            result, the contrast is computed from the data.
        """
        if self.image_layer not in self.viewer.layers:  # deleted by hand
            self._display_image(self.gco())
        data = getattr(self.gco(), f"{key}_image")
        layer = self.result_layers[key]
        if layer not in self.viewer.layers:  # not created, or removed by hand
            layer = None

        if data is None:
            if layer is not None:
                self.viewer.layers.remove(layer)
            self.result_layers[key] = None
            return

        if layer is not None:
            self.safe_update_layer(layer, data, display)
            if display is None and "gamma" in RESULT_LAYERS[key]:
                layer.gamma = RESULT_LAYERS[key]["gamma"]
            return

        settings = dict(RESULT_LAYERS[key])
        is_rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
        if is_rgb:
            settings.pop("colormap", None)
        else:
            settings.setdefault("colormap", "gray")
        layer = self._add_layer(data, rgb=is_rgb, **settings)
        if display is not None:
            self.safe_update_layer(layer, data, display)
        if key == "psf":
            layer.mouse_drag_callbacks.append(self._on_mouse_press)
        elif key in ("ee50", "fwhm", "cutoff"):
            layer.colorbar.visible = True

        # Above the image and the results before it in RESULT_LAYERS
        keys = list(RESULT_LAYERS)
        below = [self.image_layer] + [
            self.result_layers[k]
            for k in keys[: keys.index(key)]
            if self.result_layers[k] in self.viewer.layers
        ]
        dest = max(self.viewer.layers.index(lay) for lay in below) + 1
        self.viewer.layers.move(self.viewer.layers.index(layer), dest)
        self.result_layers[key] = layer

    def _save_display(self, obj):
        """Stores the contrast settings of the layers in obj"""
        layers = [("image", self.image_layer)] + [
            (key, layer)
            for key, layer in self.result_layers.items()
            if layer in self.viewer.layers
        ]
        for name, layer in layers:
            obj.display[name] = {
                "contrast_limits_range": tuple(layer.contrast_limits_range),
                "contrast_limits": tuple(layer.contrast_limits),
                "gamma": layer.gamma,
            }

    def safe_update_layer(self, layer, image, display=None):
        """
        Safely update a napari Image layer without recreating it,
        handling RGB/grayscale and dimension changes.

        Parameters
        ----------
        layer : napari.layers.Image
        image : np.ndarray
        display : dict, optional
            Contrast settings saved by _save_display. If None, the contrast
            is computed from the data.
        """

        # --- Neutral contrast, while data and rgb still match: limits meant
        # for the old data can overflow with the new dtype (e.g. 10000 with
        # uint8 data) ---
        layer.contrast_limits = (0, 1)

        # --- Detect RGB ---
        is_rgb = image.ndim == 3 and image.shape[-1] in (3, 4)

        # --- Set rgb FIRST (important) ---
        if layer.rgb != is_rgb:
            layer.rgb = is_rgb

        # --- Update data ---
        layer.data = image

        # --- Force napari to recompute dims ---
        with contextlib.suppress(AttributeError):
            layer._slice_dims(None)  # reset slicing

        with contextlib.suppress(AttributeError):
            layer._update_dims()  # recompute interpretation

        # --- Contrast: the saved one, or from the new data (e.g. a 16-bit
        # image after an 8-bit one) ---
        if display is None:
            layer.reset_contrast_limits_range()
            layer.reset_contrast_limits()
            layer.gamma = 1.0
        else:
            layer.contrast_limits_range = display["contrast_limits_range"]
            layer.contrast_limits = display["contrast_limits"]
            layer.gamma = display["gamma"]

        # --- Final refresh ---
        layer.refresh()

    def _on_image_selector_changed(self, event=None):
        if self._loading:
            return

        try:
            self.idx_active = int(self.image_selector.value)
        except (ValueError, TypeError):
            return

        self._display_current()
        self._update_widget(idx=self.idx_active)

    def _build_ui(self):
        # Creates the combobox
        self.names_layer = [
            f"{idx:03d}" for idx in range(len(self.list_deepblur))
        ]
        self.idx_active = len(self.list_deepblur) - 1
        if self.names_layer:
            self.image_selector = ComboBox(
                label="Active Image",
                choices=self.names_layer,
                value=self.names_layer[
                    0
                ],  # or use self.names_layer[self.idx_active] if you want the last one
            )
        else:
            self.image_selector = ComboBox(
                label="Active Image",
                choices=["Nothing yet"],
                value="Nothing yet",
            )
        self.image_selector.changed.connect(self._on_image_selector_changed)

        # Initialize container layout
        images_row = Container(
            widgets=[
                self.image_selector,  # dropdown added here
                self.help_button,
            ],
            layout="horizontal",
        )
        metrology_row1 = Container(
            widgets=[
                self.psf_spacing,
                self.display_psf_grid,
                self.delete_all_button,
            ],
            layout="horizontal",
        )
        metrology_row2 = Container(
            widgets=[
                self.ee50_map,
                self.fwhm_map,
                self.cutoff_map,
            ],
            layout="horizontal",
        )

        identification_model_row = Container(
            widgets=[
                self.identification_model_selection,
                self.display_model_info,
            ],
            layout="horizontal",
        )

        deblur_model_row1 = Container(
            widgets=[
                self.deblur_model_selection,
                self.deblur_action,
            ],
            layout="horizontal",
        )
        noise_row = Container(
            widgets=[
                self.gaussian_noise_level,
                self.bit_depth,
            ],
            layout="horizontal",
        )

        estimator_model_row1 = Container(
            widgets=[
                self.interpolation_model_selection,
                self.operator_spacing_x,
                self.operator_spacing_y,
            ],
            layout="horizontal",
        )

        estimator_model_row2 = Container(
            widgets=[
                self.estimate_operator,
            ],
            layout="horizontal",
        )

        self.extend(
            [
                Label(value="--- ACTIVE IMAGE SELECTION ---"),
                images_row,
                Label(value="--- IDENTIFICATION MODEL SELECTION ---"),
                identification_model_row,
                Label(value="--- METROLOGY ---"),
                metrology_row1,
                metrology_row2,
                Label(value="--- OPERATOR ESTIMATION ---"),
                estimator_model_row1,
                estimator_model_row2,
                Label(value="--- NOISE ---"),
                noise_row,
                self.identify_noise,
                Label(value="--- DEBLURRING ---"),
                deblur_model_row1,
            ]
        )

    def _update_widget(self, idx=None):
        """Activates and deactivate the actions depending on the options selected"""
        if not hasattr(self, "image_selector"):
            return
        self.names_layer = [f"{i:03d}" for i in range(len(self.list_deepblur))]

        # Choices of the magicgui widget, and selection through Qt
        self.image_selector.choices = self.names_layer
        qt_combo_box = self.image_selector.native
        if self.names_layer and 0 <= self.idx_active < len(self.names_layer):
            qt_combo_box.setCurrentText(self.names_layer[self.idx_active])
        else:
            qt_combo_box.setCurrentText(
                self.names_layer[0] if self.names_layer else "No Images"
            )

        self.deblur_action.enabled = (
            not self._busy and self.gco().physics is not None
        )

    def gco(self):
        return self.list_deepblur[self.idx_active]

    def _show_help(self, event=None):
        """Shows the help in its own window, which can stay open"""
        if self._help_dialog is None:
            self._help_dialog = QDialog(self.native)
            self._help_dialog.setWindowTitle("DeepBlur help")
            browser = QTextBrowser()
            browser.setHtml(HELP)
            QVBoxLayout(self._help_dialog).addWidget(browser)
            self._help_dialog.resize(600, 750)
        self._help_dialog.show()
        self._help_dialog.raise_()

    def _display_info(self, event):
        if self.identification_model_properties is not None:
            content = yaml.dump(
                dict(self.identification_model_properties),
                sort_keys=False,
                default_flow_style=False,
            )
            text_widget = TextEdit(value=content)
            self.viewer.window.add_dock_widget(text_widget, area="right")
        else:
            show_info("No identification model loaded yet")

    def _tiling(self):
        """Patch size of the identification model and tile spacing (y, x)"""
        patch_size = tuple(self.identification_model_properties.image_size)
        stride = (self.operator_spacing_y.value, self.operator_spacing_x.value)
        return patch_size, stride

    def _run(self, desc, compute, done=None):
        """
        Runs compute() in a worker thread, so that napari stays responsive
        (with a progress bar labelled desc), then done(result) in the main
        thread. The actions of the widget are disabled meanwhile.
        """
        if self._busy:
            show_info("DeepBlur is busy: wait for the current computation")
            return

        def errored(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # e.g. after an out of memory error
            show_error(f"{desc} failed: {e}")

        # Connected at creation, so that the errors are not raised again in
        # the main thread
        connect = {
            "errored": errored,
            "finished": lambda: self._set_busy(False),
        }
        if done is not None:
            connect["returned"] = done
        self._set_busy(True)
        # Kept alive until it finishes
        self._worker = thread_worker(
            compute, progress={"desc": desc}, connect=connect
        )()

    def _set_busy(self, busy):
        """Disables the actions while a computation runs"""
        self._busy = busy
        for widget in self._actions:
            widget.enabled = not busy
        self.deblur_action.enabled = (
            not busy and self.gco().physics is not None
        )

    def _show_if_active(self, obj, key):
        """Shows the result key of obj, if obj is still the active image"""
        if obj is self.gco():
            self._show_result(key)

    def _require_model(self):
        """
        True if an identification model is loaded. Otherwise (e.g. its
        download failed), loads it again and returns False.
        """
        if self.identification_model is not None:
            return True
        show_info("No identification model loaded yet: loading it")
        self._load_model()
        return False

    @torch.no_grad()
    def _tile_psfs(self, obj, patch_size, stride):
        """
        Returns the PSFs (1, C, K, psf_h, psf_w) estimated on the K tiles of
        obj, estimating them again only if the model or the tile spacing
        changed. Runs in a worker thread.
        """
        key = (self._model_name, stride)
        if obj.tiles_key != key:
            x = numpy_to_torch(obj.image).to(self.device)
            obj.psfs, obj.n_tiles = estimate_tiled_psfs(
                self.identification_model, x, patch_size, stride
            )
            obj.centers = tile_centers(obj.n_tiles, patch_size, stride)
            obj.tiles_key = key
        return obj.psfs

    def _display_psf_grid(self, event=None):
        """Displays the PSFs estimated on a grid of points spaced by PSF Spacing"""
        obj = self.gco()
        spacing = int(self.psf_spacing.value)
        H, W = obj.img_size
        ys, xs = np.meshgrid(
            np.arange(spacing // 2, H, spacing),
            np.arange(spacing // 2, W, spacing),
            indexing="ij",
        )
        centers = np.stack([ys.ravel(), xs.ravel()], axis=1)
        self._display_psfs_at(obj, centers, "Estimating the PSF grid")

    def _display_psfs_at(self, obj, centers, desc):
        """
        Adds to the psf layer of obj the PSFs estimated on the patches
        centered on centers (N, 2), as (y, x).
        """
        if not self._require_model():
            return
        model = self.identification_model
        patch_size = tuple(self.identification_model_properties.image_size)

        @torch.no_grad()
        def compute():
            x = self.data_normalization_fn(
                numpy_to_torch(obj.image).to(self.device)
            )
            psf_image = estimate_psfs_at_points(
                model, x, torch.as_tensor(centers), patch_size
            )
            return torch_to_numpy(psf_image.mean(dim=1, keepdim=True))

        def done(psf_image):
            # Over the PSFs already shown (grid, clicks or operator)
            if obj.psf_image is not None:
                psf_image = np.maximum(obj.psf_image, psf_image)
            obj.psf_image = psf_image
            self._show_if_active(obj, "psf")

        self._run(desc, compute, done)

    def _display_ee50_map(self, event=None):
        """
        Displays the EE50 of the PSFs estimated on the tiles: the radius of
        the smallest disk that contains 50% of the PSF energy (L1 norm), with
        an optimized center.
        """
        self._display_tile_map(
            "ee50",
            "EE50 radius",
            lambda psfs: encircled_energy_radius(psfs)[0],
        )

    def _display_fwhm_map(self, event=None):
        """
        Displays the FWHM of the PSFs estimated on the tiles, for PSFs of any
        shape: the diameter of the disk with the area of the region where the
        PSF exceeds half its maximum (see full_width_half_maximum).
        """
        self._display_tile_map("fwhm", "FWHM", full_width_half_maximum)

    def _display_cutoff_map(self, event=None):
        """
        Displays the cut-off frequency of the PSFs estimated on the tiles,
        in cycles per pixel: the radius of the disk with the area of the
        frequencies where the MTF exceeds 5% (see cutoff_frequency).
        """
        self._display_tile_map(
            "cutoff",
            "Cut-off frequency",
            cutoff_frequency,
            unit="cycles/px (Nyquist: 0.5)",
        )

    def _display_tile_map(self, key, name, metric, unit="px"):
        """
        Displays in the layer `key` a metric of the PSFs estimated on the
        tiles, bilinearly interpolated between the tile centers.

        Parameters
        ----------
        key : str
            Result of DeepBlurObject, see RESULT_LAYERS.
        name : str
            Name of the metric, in the messages.
        metric : callable
            Maps PSFs (..., h, w) to values (...).
        unit : str
            Unit of the values, in the messages.
        """
        if not self._require_model():
            return
        obj = self.gco()
        patch_size, stride = self._tiling()

        def compute():
            psfs = self._tile_psfs(obj, patch_size, stride)
            _, C, _, psf_h, psf_w = psfs.shape
            psfs = psfs.reshape(1, C, *obj.n_tiles, psf_h, psf_w)

            # The tiles reaching the zero padding at the bottom and right of
            # the image bias the PSFs: keep the ones inside the image, if any
            H, W = obj.img_size
            n_rows = max(1, (H - patch_size[0]) // stride[0] + 1)
            n_cols = max(1, (W - patch_size[1]) // stride[1] + 1)
            values = metric(psfs[:, :, :n_rows, :n_cols])

            image = interpolate_tile_map(values, patch_size, stride, (H, W))
            # One map for all channels (they share the image plane)
            image = torch_to_numpy(image.mean(dim=1, keepdim=True))
            return image, values, (psf_h, psf_w)

        def done(result):
            image, values, (psf_h, psf_w) = result
            setattr(obj, f"{key}_image", image)
            self._show_if_active(obj, key)
            per_channel = ", ".join(
                f"{float(v.median()):.3g}" for v in values[0].flatten(1)
            )
            show_info(
                f"{name}: {float(values.min()):.3g} to "
                f"{float(values.max()):.3g}, median {per_channel} {unit} "
                f"(PSFs of {psf_h}x{psf_w} px)"
            )

        self._run(f"Computing the {name} map", compute, done)

    def _save_results_callback(self, event):
        print("Save results!")

    def _delete_current_object(self, event):
        if self.idx_active is None or self.idx_active >= len(
            self.list_deepblur
        ):
            print("No active object to delete.")
            return

        # The widget always needs an active image (see gco)
        if len(self.list_deepblur) == 1:
            show_info("Cannot delete the last image.")
            return

        # Remove from list
        del self.list_deepblur[self.idx_active]
        print(f"Deleted DeepBlur object at index {self.idx_active}.")

        # Select a new active image. The selector callback is muted while
        # its choices are rebuilt, otherwise it would reset idx_active.
        self.idx_active = min(self.idx_active, len(self.list_deepblur) - 1)
        self._loading = True
        try:
            self._update_widget()
            self._display_current()
        finally:
            self._loading = False

    def _delete_all_callback(self, event):
        cur_obj = self.gco()
        cur_obj.psf_image = None
        self._show_result("psf")

        print("Delete all!")

    def _deblur_action(self, event=None):
        obj = self.gco()
        if obj.physics is None:
            show_info("Estimate an operator first")
            return

        # Noise model of the gradio app and of the noise identification
        # network, in photon counts: Poisson(G x) + N(0, sigma^2), with
        # G = 2**bit_depth - 1. In the [0, 1] scale of y, RAM takes
        # gain = 1/G and sigma = sigma/G.
        gain, sigma = to_ram_param(
            2**self.bit_depth.value - 1,
            self.gaussian_noise_level.value,
        )
        model_name = self.deblur_model_selection.value

        @torch.no_grad()
        def compute():
            y = numpy_to_torch(obj.image).to(
                device=self.device, dtype=self.dtype
            )
            y = self.data_normalization_fn(y)
            x_est = self._deblur(model_name, y, obj.physics, gain, sigma)

            # The operators are "valid" convolutions: x_est is larger than y
            # by the PSF size of the operator minus 1
            h, w = obj.operator_psf_size
            x_est = x_est[..., h // 2 : -(h // 2), w // 2 : -(w // 2)]
            return torch_to_numpy(x_est, idx=0)

        def done(deblur_image):
            obj.deblur_image = deblur_image
            self._show_if_active(obj, "deblur")
            show_info("Deblurring done")

        self._run(f"Deblurring ({model_name})", compute, done)

    def _deblur(self, model_name, y, physics, gain, sigma):
        """Deblurs y with the method model_name. Runs in a worker thread."""
        # Methods modelling Gaussian noise only use the noise std at the
        # mean intensity
        sigma_gauss = float(torch.sqrt(y.mean() * gain + sigma**2))

        if model_name == "Tikhonov":
            return physics.prox_l2(
                physics.A_adjoint(y),
                y,
                1 / sigma_gauss**2,
                max_iter=30,
            )

        if model_name == "TV":
            # Prior with fewer iterations than the default, for speed
            prior = dinv.optim.prior.TVPrior(n_it_max=20)
            model = dinv.optim.optimizers.optim_builder(
                iteration="PGD",
                prior=prior,
                data_fidelity=dinv.optim.data_fidelity.L2(),
                early_stop=True,
                max_iter=300,
                verbose=False,
                # lambda: TV regularisation parameter
                params_algo={"stepsize": 1.0, "lambda": 10 * sigma_gauss**2},
            )
            return model(y, physics)

        if model_name == "RAM":
            deblur_model = self._get_ram_model(y.shape[1])
            return deblur_model(y, physics, sigma=sigma, gain=gain)

        if model_name == "DPIR":
            prior = dinv.optim.prior.PnP(
                denoiser=self._get_deblur_model("DRUNet")
            )
            deblur_model = dinv.optim.optim_builder(
                iteration="PGD",
                prior=prior,
                data_fidelity=dinv.optim.data_fidelity.L2(),
                params_algo={"stepsize": 1.0, "g_param": sigma_gauss},
                max_iter=20,
            )
            # DRUNet denoises color images
            if y.shape[1] == 1:
                return deblur_model(y.expand(-1, 3, -1, -1), physics)[:, :1]
            return deblur_model(y, physics)

        raise ValueError(f"Unknown deblurring method: {model_name}")

    def _get_deblur_model(self, name):
        """Returns a deblurring network, loading it on first use."""
        if name not in self._deblur_models:
            show_info(f"Loading {name} weights...")
            if name == "DRUNet":
                model = dinv.models.DRUNet(
                    pretrained="download", device=self.device
                )
            else:
                raise ValueError(f"Unknown deblurring network: {name}")
            self._deblur_models[name] = model.eval()
        return self._deblur_models[name]

    def _get_ram_model(self, n_channels):
        """
        Returns the RAM fine-tuned for grayscale (1 channel) or colour images,
        loading it on first use. Falls back to the deepinv RAM weights if the
        fine-tuned weights can not be downloaded.
        """
        n_channels = 1 if n_channels == 1 else 3
        key = ("RAM", n_channels)
        if key not in self._deblur_models:
            show_info("Loading RAM weights...")
            try:
                ckpt_path = get_weights(RAM_CHECKPOINTS[n_channels])
            except WeightsUnavailableError as e:
                show_warning(f"{e}. Using the deepinv RAM weights instead.")
                model = dinv.models.RAM(pretrained=True, device=self.device)
            else:
                model = dinv.models.RAM(pretrained=False, device=self.device)
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=self.device)
                )
            self._deblur_models[key] = model.to(self.device).eval()
        return self._deblur_models[key]

    def _get_noise_id_model(self):
        """
        Returns the noise identification network, loading it on first use.
        Raises WeightsUnavailableError if its weights can not be downloaded.
        """
        if self._noise_id_model is None:
            show_info("Loading noise identification weights...")
            ckpt_path = get_weights(NOISE_ID_CHECKPOINT)
            model = NoiseLevelIdNet()
            model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device)
            )
            self._noise_id_model = model.to(self.device).eval()
        return self._noise_id_model

    def _identify_noise(self, event=None):
        """Estimates the noise parameters of the active image"""
        image = self.gco().image

        @torch.no_grad()
        def compute():
            model = self._get_noise_id_model()
            y = numpy_to_torch(image).to(self.device)
            y = self.data_normalization_fn(y)
            # The network sees each channel as a separate grayscale image
            y = y.reshape(-1, 1, *y.shape[-2:])
            est = model(y)  # (C, 2): sigma and gain, in photon counts
            return float(est[:, 0].mean()), float(est[:, 1].mean())

        def done(result):
            sigma, gain = result
            # Bit depth first, as it sets the maximum of sigma (the gradio
            # app clips sigma to 5)
            self.bit_depth.value = int(
                np.clip(round(np.log2(gain + 1)), 2, 16)
            )
            self.gaussian_noise_level.value = float(
                np.clip(sigma, 0.0, self.gaussian_noise_level.max)
            )
            show_info(
                f"Noise identified: σ = {sigma:.3f}, gain = {gain:.1f} "
                f"→ bit depth {self.bit_depth.value}"
            )

        self._run("Estimating the noise", compute, done)

    def _on_bit_depth_changed(self, event=None):
        """
        Sigma is in photon counts, and the image spans 2**bit_depth - 1
        counts: its maximum follows the bit depth
        """
        self.gaussian_noise_level.max = 2**self.bit_depth.value - 1

    def _load_model(self, event=None):
        """Loads the selected identification model, in a worker thread"""
        name = self.identification_model_selection.value
        self.identification_model = None
        self.identification_model_properties = None
        self._model_name = None

        def compute():
            return load_id_model(name, device=self.device)

        def done(result):
            model, properties = result
            self.identification_model = model
            self.identification_model_properties = properties
            self._model_name = name
            if properties.weights_source == "huggingface":
                show_warning(
                    f"Weights server unreachable: model '{name}' comes from "
                    "the Hugging Face hub, which may have an older version"
                )
            else:
                show_info(f"Model '{name}' loaded")

        self._run(f"Loading the model {name}", compute, done)

    def _on_mouse_press(self, layer, event):
        """Displays the PSF estimated around the clicked point"""
        if event.button != 1:
            return
        position = self.viewer.cursor.position
        if position is None:
            return
        obj = self.gco()
        point = [int(round(p)) for p in position[-2:]]  # (y, x)
        if all(0 <= i < n for i, n in zip(point, obj.img_size, strict=True)):
            self._display_psfs_at(obj, np.array([point]), "Estimating the PSF")

    def _estimate_operator(self, event=None):
        if not self._require_model():
            return
        obj = self.gco()
        patch_size, stride = self._tiling()
        mode = self.interpolation_model_selection.value

        # One PSF for an image smaller than a patch, and eigen PSFs need a 2D
        # grid of tiles (thin plate splines are singular on a line)
        img_h, img_w = obj.img_size
        if img_h < patch_size[0] or img_w < patch_size[1]:
            if mode != "constant":
                show_info(
                    f"The image is smaller than the {patch_size[0]}x"
                    f"{patch_size[1]} patches of the model: constant operator"
                )
            mode = "constant"
        elif mode == "eigen_psf" and (
            img_h == patch_size[0] or img_w == patch_size[1]
        ):
            show_info("A single row of tiles: tiled_psf operator")
            mode = "tiled_psf"

        @torch.no_grad()
        def compute():
            psfs = self._tile_psfs(obj, patch_size, stride)
            psf_h, psf_w = psfs.shape[-2:]
            if mode == "tiled_psf":
                physics = TiledSpaceVaryingBlur(
                    filters=psfs,
                    patch_size=(
                        patch_size[0] + psf_h - 1,
                        patch_size[1] + psf_w - 1,
                    ),
                    stride=stride,
                    blending_mode="bump",
                    use_fft=True,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif mode == "eigen_psf":
                physics = build_space_varying_operator(
                    psfs, obj.centers, obj.img_size, self.device
                )
            else:
                physics = constant_operator(
                    psfs, obj.centers, obj.img_size, patch_size
                ).to(self.device)

            # PSFs of the operator on a grid of Diracs, in grayscale
            grid = torch.zeros(
                1,
                obj.nchannels,
                img_h + psf_h - 1,
                img_w + psf_w - 1,
                device=self.device,
                dtype=self.dtype,
            )
            grid[
                ..., stride[0] // 2 :: stride[0], stride[1] // 2 :: stride[1]
            ] = 1
            psf_image = physics(grid).mean(dim=1, keepdim=True)
            return physics, (psf_h, psf_w), torch_to_numpy(psf_image)

        def done(result):
            obj.physics, obj.operator_psf_size, obj.psf_image = result
            self._show_if_active(obj, "psf")
            show_info(f"Operator estimated ({mode})")

        self._run("Estimating the operator", compute, done)


def constant_operator(psfs, centers, img_size, patch_size):
    """
    Convolution by the medoid of the PSFs of the tiles (per channel): the
    PSF with the smallest total L1 distance to the others.

    Parameters
    ----------
    psfs : torch.Tensor
        PSFs (1, C, K, psf_h, psf_w) of the K tiles
    centers : torch.Tensor
        tile centers (K, 2)
    img_size, patch_size : tuple
        (H, W) of the image and of the tiles
    """
    psfs = psfs[0]  # (C, K, psf_h, psf_w)
    n_channels = psfs.shape[0]
    d0 = patch_size[0] // 2 - 1
    d1 = patch_size[1] // 2 - 1

    # Keep only the tiles lying fully inside the image (not in padding)
    cc = centers
    mask = (
        (cc[:, 0] >= d0)
        & (cc[:, 1] >= d1)
        & (cc[:, 0] <= img_size[0] - d0)
        & (cc[:, 1] <= img_size[1] - d1)
    ).to(psfs.device)
    if not mask.any():
        # Image smaller than one patch: every tile touches the padding
        mask[:] = True
    psfs_filtered = psfs[:, mask]  # (C, N, psf_h, psf_w)
    psfs_flat = psfs_filtered.flatten(-2)  # (C, N, psf_h*psf_w)

    total_distances = torch.cdist(psfs_flat, psfs_flat, p=1).sum(-1)
    medoid_idx = total_distances.argmin(dim=-1)  # (C,)
    medoid_psf = psfs_filtered[
        torch.arange(n_channels, device=psfs.device), medoid_idx
    ]  # (C, psf_h, psf_w)
    return dinv.physics.Blur(filter=medoid_psf[None], padding="valid")


def to_2d_image(data, rgb):
    """
    Returns data as a 2D grayscale (H, W) or RGB (H, W, 3) array, dropping
    the alpha channel, or None if it is not a 2D image (e.g. a z-stack).

    Parameters
    ----------
    data : array-like
    rgb : bool
        Whether the last axis of data holds colors
    """
    image = np.asarray(data)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    elif rgb and image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim == 2 or (rgb and image.ndim == 3 and image.shape[-1] == 3):
        return image
    return None


def _layer_image(layer):
    """The data of a napari image layer as in to_2d_image (a copy)"""
    data = layer.data[0] if layer.multiscale else layer.data
    image = to_2d_image(data, layer.rgb)
    return None if image is None else np.array(image)


class DeepBlurObject:
    def __init__(self, image):
        self.nchannels = 1 if image.ndim == 2 else image.shape[-1]
        self.image = image
        self.img_size = self.image.shape[:2]

        # Results, None until computed (see DeepBlurWidget._show_result)
        self.operator_psf_size = None  # PSF size of physics
        self.deblur_image = None
        self.psf_image = None

        self.physics = None

        # PSFs estimated on the tiles (see DeepBlurWidget._tile_psfs)
        self.psfs = None
        self.n_tiles = None
        self.centers = None
        self.tiles_key = None
        self.ee50_image = None
        self.fwhm_image = None
        self.cutoff_image = None

        # Contrast settings of each layer, saved when switching image
        self.display = {}


def build_space_varying_operator(
    psfs, centers, output_img_size, device, n_eigen=10
):
    """
    Build a SpaceVaryingBlur operator from measured PSFs and their centers.

    Args:
        psfs: tensor of shape (1, C, N, psf_h, psf_w) — N measured PSFs
        centers: tensor of shape (N, 2) — pixel coordinates of PSF centers
        output_img_sizeze: tuple (H, W) — full image size as an output. So input will be (H+h, W+w), where (h,w)=psf_size
        device: torch device

    Returns:
        SpaceVaryingBlur operator
    """

    B, C, N, psf_h, psf_w = psfs.shape

    H, W = output_img_size
    H_pad = H + psf_h - 1
    W_pad = W + psf_w - 1

    # Normaliser les centres en [0, 1]
    centers_shifted = centers + torch.tensor(
        [psf_h // 2, psf_w // 2], device=centers.device
    )
    centers_norm = centers_shifted / torch.tensor(
        [H_pad, W_pad], dtype=torch.float32, device=centers.device
    )

    # SVD pour calculer les eigen-PSFs
    psf_flat = psfs.flatten(-2, -1)  # (B, C, N, psf_h*psf_w)
    _, _, V = torch.linalg.svd(psf_flat, full_matrices=False)
    # V : (B, C, N, psf_h*psf_w)

    n_eigen = min(N, n_eigen)
    V = V[..., :n_eigen, :]  # (B, C, n_eigen, psf_h*psf_w)
    eigen_psf = V.reshape(B, C, n_eigen, psf_h, psf_w)

    # Coefficients de projection
    coeffs = torch.matmul(psf_flat, V.transpose(-1, -2))  # (B, C, N, n_eigen)

    # Interpolation thin plate spline sur toute l'image
    T0 = torch.linspace(0, 1, H_pad, device=device)
    T1 = torch.linspace(0, 1, W_pad, device=device)
    yy, xx = torch.meshgrid(T0, T1, indexing="ij")
    XX = torch.stack((yy.flatten(), xx.flatten()), dim=1)  # (H*W, 2)

    tps = ThinPlateSpline(0.0)
    tps.fit(centers_norm, coeffs)  # fit sur les N centres
    w = tps.transform(XX).transpose(-1, -2)  # (B, C, n_eigen, H*W)
    w = w.reshape(B, C, n_eigen, H_pad, W_pad)

    # Construire l'opérateur
    physics = SpaceVaryingBlur(
        filters=eigen_psf,
        multipliers=w,
        device=device,
        padding="valid",
        use_fft=True,
    )
    return physics
