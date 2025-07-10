import numpy as np

import genesis as gs

########################## 初期化 ##########################
gs.init(backend=gs.gpu)

########################## シーンの作成 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = False,
)

########################## エンティティ ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=np.array([0, 0, 0.32]),
    ),
)
floating_camera = scene.add_camera(
    pos=np.array([-2, -2, 2]),
    lookat=np.array([0, 0, 0]),
    # res=(720, 480),
    fov=40,
    GUI=False,
)
########################## ビルド ##########################
scene.build()

jnt_names = [
    'FR_hip_joint',
    'FR_thigh_joint',
    'FR_calf_joint',
    'FL_hip_joint',
    'FL_thigh_joint',
    'FL_calf_joint',
    'RR_hip_joint',
    'RR_thigh_joint',
    'RR_calf_joint',
    'RL_hip_joint',
    'RL_thigh_joint',
    'RL_calf_joint',
]
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

############ オプション：制御ゲインの設定 ############
# 位置ゲインの設定
robot.set_dofs_kp(
    kp             = np.array([70,70,70,70,70,70,70,70,70,70,70,70,]),
    dofs_idx_local = dofs_idx,
)
# 速度ゲインの設定
robot.set_dofs_kv(
    kv             = np.array([3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,]),
    dofs_idx_local = dofs_idx,
)

robot.set_dofs_position(
    np.array([
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
    ]),
    dofs_idx,
)
floating_camera.start_recording()
# PD制御
for i in range(500):
    # robot_pos = np.array(robot.get_pos()[0].cpu())
    # floating_camera.set_pose(pos=robot_pos + np.array([-1, -1, 1]), lookat=robot_pos + np.array([0, 0, -0.1]))
    if i == 0:
        robot.control_dofs_position(
            np.array([
                0.0, 0.8, -1.5,
                0.0, 0.8, -1.5,
                0.0, 0.8, -1.5,
                0.0, 0.8, -1.5,
            ]),
            dofs_idx,
        )
    elif i == 250:
        robot.control_dofs_position(
            np.array([
                0.0, 0.8, -1.5,
                0.0, 0.8, -1.5,
                0.0, 1.8, -0.8,
                0.0, 1.8, -0.8,
            ]),
            dofs_idx,
        )
    # これは与えられた制御コマンドに基づいて計算された制御力です
    # 力制御を使用している場合、これは与えられた制御コマンドと同じです
    print('control force:', robot.get_dofs_control_force(dofs_idx))

    # これは自由度が実際に経験している力です 
    print('internal force:', robot.get_dofs_force(dofs_idx))

    scene.step()
    floating_camera.render()
floating_camera.stop_recording("robot_test.mp4", fps = 100)
