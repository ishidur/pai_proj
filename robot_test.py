import genesis as gs
import numpy as np
import torch

########################## 初期化 ##########################
gs.init(backend=gs.gpu)

########################## シーンの作成 ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, 3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

bucket_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=gs.device)
bucket_pos = torch.tensor([0, 0, 0.5], dtype=torch.float, device=gs.device)

########################## エンティティ ##########################
bucket = scene.add_entity(
    gs.morphs.URDF(
        file="./assets/zx120/zx120-bucket.urdf",
        collision=False,
    ),
)

########################## ビルド ##########################
scene.build()


while True:
    bucket.set_pos(bucket_pos, zero_velocity=True)
    bucket.set_quat(bucket_quat, zero_velocity=True)
    scene.step()
