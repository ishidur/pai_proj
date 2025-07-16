import genesis as gs
import numpy as np
from itertools import chain

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(20, 20, 20),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        res=(640, 640),
    ),
    show_viewer=True,
)
plane = scene.add_entity(gs.morphs.Plane())
excavator = scene.add_entity(
    # gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    gs.morphs.URDF(file="./assets/zx120/zx120.urdf", pos=[0, 0, 0]),
)

scene.build()

jnt_names = [
    # "root_joint",
    "swing_joint",
    "boom_joint",
    "arm_joint",
    "bucket_joint",
    "bucket_end_joint",
]
dofs_idx = list(
    chain.from_iterable(
        [excavator.get_joint(name).dofs_idx_local for name in jnt_names]
    )
)
print(dofs_idx)
# set force range for safety
excavator.set_dofs_force_range(
    lower=np.array([-100, -100, -100, -100, -100]),
    upper=np.array([100, 100, 100, 100, 100]),
    dofs_idx_local=dofs_idx,
)

for i in range(3000):
    if i == 0:
        excavator.control_dofs_position(np.array([0, 0, 0, 0, 0]), dofs_idx)
    elif i == 250:
        excavator.control_dofs_position(np.array([np.pi / 2, 0, 0, 0, 0]), dofs_idx)
    elif i == 500:
        excavator.control_dofs_position(np.array([np.pi * 3 / 2, 0, 0, 0, 0]), dofs_idx)
    elif i == 750:
        excavator.control_dofs_position(
            np.array([np.pi * 3 / 2, -1.22, 0, 0, 0]), dofs_idx
        )
    elif i == 1000:
        excavator.control_dofs_position(
            np.array([np.pi * 3 / 2, 0.76, 0, 0, 0]), dofs_idx
        )
    elif i == 1250:
        excavator.control_dofs_position(np.array([np.pi * 3 / 2, 0, 0, 0, 0]), dofs_idx)
    elif i == 1500:
        excavator.control_dofs_position(
            np.array([np.pi * 3 / 2, 0, 0.5, 0, 0]), dofs_idx
        )
    elif i == 1750:
        excavator.control_dofs_position(
            np.array([np.pi * 3 / 2, 0, 2.6, 0, 0]), dofs_idx
        )
    elif i == 2000:
        excavator.control_dofs_position(np.array([np.pi * 3 / 2, 0, 0, 0, 0]), dofs_idx)
    elif i == 2250:
        excavator.control_dofs_position(
            np.array([np.pi * 3 / 2, 0, 0, -0.53, 0]), dofs_idx
        )
    elif i == 2500:
        excavator.control_dofs_position(
            np.array([np.pi * 3 / 2, 0, 0, 2.49, 0]), dofs_idx
        )
    elif i == 2750:
        excavator.control_dofs_position(np.array([np.pi * 3 / 2, 0, 0, 0, 0]), dofs_idx)

    scene.step()
