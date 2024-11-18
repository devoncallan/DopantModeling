import imageio
import numpy as np

import napari
from napari_animation import Animation

ORTHO_ANGLE = (90, 35.264, 135)
ORTHO_ZOOM = 1.5

X_ROTATE = (0, 0, 90)

def add(tuple1, tuple2):
    return tuple(x + y for x, y in zip(tuple1, tuple2))


def create_napari_viewer(image_3D:np.ndarray):

    viewer = napari.Viewer(ndisplay=3)
    layer = viewer.add_image(image_3D)

    layer.blending = 'translucent_no_depth'
    layer.interpolation = 'cubic'
    layer.colormap = 'twilight_shifted'
    layer.rendering = 'attenuated_mip'
    layer.attenuation = 0.03

    viewer.camera.angles = ORTHO_ANGLE
    viewer.camera.zoom = ORTHO_ZOOM

    return viewer, layer

def create_rotate_animation(viewer, layer):


    animation = Animation(viewer)
    num_steps = 120

    animation.capture_keyframe()
    current_angle = add(ORTHO_ANGLE, X_ROTATE)
    viewer.camera.angles = current_angle
    animation.capture_keyframe(steps=num_steps)

    current_angle = add(current_angle, X_ROTATE)
    viewer.camera.angles = current_angle
    animation.capture_keyframe(steps=num_steps)

    current_angle = add(current_angle, X_ROTATE)
    viewer.camera.angles = current_angle
    animation.capture_keyframe(steps=num_steps)

    current_angle = add(current_angle, X_ROTATE)
    viewer.camera.angles = current_angle
    animation.capture_keyframe(steps=num_steps)

    animation.animate('morphology.gif', canvas_only=True, fps=60, quality=9)

    napari.run()

