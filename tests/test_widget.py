"""
Tests of the widget in a headless napari viewer. They use the identification
weights, downloaded once from the MAMBO server (see _weights.py).
"""

import numpy as np
import pytest
from qtpy.QtWidgets import QTextBrowser

from deepblur_napari._widget import DEFAULT_IMAGES, DeepBlurWidget

TIMEOUT = 600_000  # ms, for the first downloads of the weights


@pytest.fixture
def widget(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = DeepBlurWidget(viewer)
    wait(qtbot, widget)
    assert widget.identification_model is not None
    yield widget
    wait(qtbot, widget)


def wait(qtbot, widget):
    """Waits for the computation of the widget, if any"""
    qtbot.waitUntil(lambda: not widget._busy, timeout=TIMEOUT)


def run(qtbot, widget, action, *args):
    action(*args)
    wait(qtbot, widget)


def add_image(qtbot, widget, data, **kwargs):
    """Adds a layer to the viewer, as a user would, and waits for the widget"""
    widget.viewer.add_image(data, **kwargs)
    qtbot.waitUntil(lambda: not widget._loading, timeout=5000)


def names(widget):
    return [layer.name for layer in widget.viewer.layers]


def test_opens_with_the_demo_images(widget):
    assert len(widget.list_deepblur) == len(DEFAULT_IMAGES)
    assert names(widget) == ["image"]


def test_leaves_other_layers_alone(widget, qtbot):
    n = len(widget.list_deepblur)
    widget.viewer.add_points(np.array([[10, 10], [50, 60]]))
    widget.viewer.add_labels(np.zeros((200, 200), dtype=np.int32))
    qtbot.wait(100)
    assert len(widget.list_deepblur) == n
    assert names(widget) == ["image", "Points", "Labels"]


def test_takes_2d_images_only(widget, qtbot):
    n = len(widget.list_deepblur)
    widget.viewer.add_image(np.random.rand(10, 64, 64), name="stack")
    qtbot.wait(100)
    assert len(widget.list_deepblur) == n
    assert "stack" in names(widget)

    rgba = np.random.randint(0, 255, (300, 320, 4), dtype=np.uint8)
    add_image(qtbot, widget, rgba, name="rgba")
    assert len(widget.list_deepblur) == n + 1
    assert widget.gco().image.shape == (300, 320, 3)
    assert "rgba" not in names(widget)


@pytest.mark.parametrize("operator", ["constant", "tiled_psf", "eigen_psf"])
def test_operators_and_deblurring(widget, qtbot, operator):
    widget.interpolation_model_selection.value = operator
    widget.deblur_model_selection.value = "Tikhonov"
    run(qtbot, widget, widget._estimate_operator)
    run(qtbot, widget, widget._deblur_action)

    obj = widget.gco()
    assert obj.physics is not None
    assert obj.deblur_image.shape == obj.image.shape
    assert np.isfinite(obj.deblur_image).all()
    assert names(widget) == ["image", "deblur", "psf"]


def test_ram_on_a_color_image(widget, qtbot):
    image = np.random.randint(0, 255, (330, 340, 3), dtype=np.uint8)
    add_image(qtbot, widget, image, name="color")
    widget.interpolation_model_selection.value = "tiled_psf"
    widget.deblur_model_selection.value = "RAM"
    run(qtbot, widget, widget._estimate_operator)
    run(qtbot, widget, widget._deblur_action)
    assert widget.gco().deblur_image.shape == image.shape


def test_image_smaller_than_a_patch(widget, qtbot):
    add_image(qtbot, widget, np.random.rand(100, 120).astype(np.float32))
    widget.interpolation_model_selection.value = "eigen_psf"
    widget.deblur_model_selection.value = "Tikhonov"
    run(qtbot, widget, widget._estimate_operator)
    run(qtbot, widget, widget._deblur_action)
    run(qtbot, widget, widget._display_ee50_map)
    run(qtbot, widget, widget._display_fwhm_map)

    obj = widget.gco()
    assert obj.physics.__class__.__name__ == "Blur"  # constant operator
    assert obj.deblur_image.shape == (100, 120)
    assert obj.ee50_image.shape == (100, 120)
    assert obj.fwhm_image.shape == (100, 120)


def test_single_row_of_tiles(widget, qtbot):
    # As high as a patch of z3_z36: eigen PSFs fall back to tiled PSFs
    add_image(qtbot, widget, np.random.rand(320, 600).astype(np.float32))
    widget.interpolation_model_selection.value = "eigen_psf"
    run(qtbot, widget, widget._estimate_operator)
    assert widget.gco().physics.__class__.__name__ == "TiledSpaceVaryingBlur"


def test_model_change_keeps_the_deblurred_image_aligned(widget, qtbot):
    widget.interpolation_model_selection.value = "tiled_psf"
    widget.deblur_model_selection.value = "Tikhonov"
    run(qtbot, widget, widget._estimate_operator)  # z3_z36: 71x71 PSFs

    widget.identification_model_selection.value = "z3_z10"  # 51x51 PSFs
    wait(qtbot, widget)
    run(qtbot, widget, widget._deblur_action)
    assert widget.gco().deblur_image.shape == widget.gco().image.shape


def test_maps_and_noise(widget, qtbot):
    run(qtbot, widget, widget._display_ee50_map)
    run(qtbot, widget, widget._display_fwhm_map)
    run(qtbot, widget, widget._display_cutoff_map)
    obj = widget.gco()
    for image in (obj.ee50_image, obj.fwhm_image, obj.cutoff_image):
        assert image.shape == obj.image.shape[:2]
        assert (image > 0).all()
    assert (obj.cutoff_image <= 0.5).all()
    assert names(widget) == [
        "image",
        "EE50 radius (px)",
        "FWHM (px)",
        "Cut-off (cycles/px)",
    ]

    run(qtbot, widget, widget._identify_noise)
    assert widget.gaussian_noise_level.value > 0
    assert widget.gaussian_noise_level.max == 2**widget.bit_depth.value - 1


def test_layers_follow_the_active_image(widget, qtbot):
    start = widget.idx_active
    run(qtbot, widget, widget._estimate_operator)
    run(qtbot, widget, widget._display_ee50_map)
    assert names(widget) == ["image", "psf", "EE50 radius (px)"]

    widget.image_selector.value = "000"
    assert names(widget) == ["image"]
    widget.image_selector.value = f"{start:03d}"
    assert names(widget) == ["image", "psf", "EE50 radius (px)"]

    widget._delete_all_callback(None)
    assert names(widget) == ["image", "EE50 radius (px)"]


def test_deleted_image_layer_comes_back(widget, qtbot):
    widget.viewer.layers.remove(widget.image_layer)
    run(qtbot, widget, widget._estimate_operator)
    assert names(widget) == ["image", "psf"]
    widget.image_selector.value = "000"
    assert names(widget)[0] == "image"


def test_psf_grid_and_clicked_psfs(widget, qtbot):
    widget.psf_spacing.value = 128
    run(qtbot, widget, widget._display_psf_grid)
    obj = widget.gco()
    assert obj.psf_image.shape == obj.image.shape[:2]
    assert names(widget) == ["image", "psf"]

    # A clicked PSF adds to the grid
    grid = obj.psf_image.copy()
    run(qtbot, widget, widget._display_psfs_at, obj, np.array([[50, 60]]), "")
    assert (obj.psf_image >= grid).all() and (obj.psf_image > grid).any()

    widget._delete_all_callback(None)
    assert obj.psf_image is None and names(widget) == ["image"]


def test_help(widget):
    widget._show_help()
    assert widget._help_dialog.isVisible()
    text = widget._help_dialog.findChild(QTextBrowser).toPlainText()
    assert "EE50 Map" in text and "tiled_psf" in text
    widget._help_dialog.close()
