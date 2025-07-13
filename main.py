import genesis as gs
import numpy as np

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(20, 20, 20),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        res=(640, 640),
    ),
    show_viewer=False,
)
plane = scene.add_entity(gs.morphs.Plane())
excavator = scene.add_entity(
    # gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    gs.morphs.URDF(file="./assets/zx120/zx120.urdf", pos=[0, 0, 0]),
)

floating_camera = scene.add_camera(
    pos=np.array([20, 20, 20]),
    lookat=np.array([0, 0, 0]),
    res=(640, 640),
    fov=40,
    GUI=False,
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
dofs_idx = [excavator.get_joint(name).dof_idx_local for name in jnt_names]
print(dofs_idx)
# set positional gains
excavator.set_dofs_kp(
    kp=np.array([4500, 3500, 3500, 2000, 2000]),
    dofs_idx_local=dofs_idx,
)
# set velocity gains
excavator.set_dofs_kv(
    kv=np.array([450, 350, 350, 200, 200]),
    dofs_idx_local=dofs_idx,
)
# set force range for safety
excavator.set_dofs_force_range(
    lower=np.array([-100, -100, -100, -100, -100]),
    upper=np.array([100, 100, 100, 100, 100]),
    dofs_idx_local=dofs_idx,
)
floating_camera.start_recording()

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
    floating_camera.render()

floating_camera.stop_recording("excavator_move_test.mp4", fps = 60)
