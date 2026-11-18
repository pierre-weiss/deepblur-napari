# Opens napari with the DeepBlur widget: uv run python debug.py
import napari
from deepblur_napari._widget import DeepBlurWidget

viewer = napari.Viewer()
widget = DeepBlurWidget(viewer)
viewer.window.add_dock_widget(widget, area="right")

napari.run()
