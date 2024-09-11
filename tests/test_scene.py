import torch
import numpy as np

from nndepth.scene import Frame, Depth, Disparity


def get_frame():
    image = torch.ones((3, 480, 640))
    depth = torch.ones((1, image.shape[-2], image.shape[-1]))
    depth = Depth(data=depth)
    disparity = torch.ones((1, image.shape[-2], image.shape[-1]))
    disparity = Disparity(data=disparity, disp_sign="negative")
    cam_intrinsic = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1)
    cam_ext = torch.eye(4)
    pose = torch.eye(4)
    frame = Frame(
        image=torch.Tensor(image),
        disparity=disparity,
        planar_depth=depth,
        cam_intrinsic=cam_intrinsic,
        cam_extrinsic=cam_ext,
        pose=pose
    )
    return frame


def test_frame_batching():
    frame = get_frame()
    new_frame = frame.unsqueeze(0)
    assert new_frame.batch_size[0] == 1
    assert new_frame.image.shape[0] == 1
    assert new_frame.disparity.batch_size[0] == 1
    assert new_frame.planar_depth.batch_size[0] == 1


def test_frame_resize():
    frame = get_frame()
    new_frame = frame.resize((384, 384))
    assert new_frame.image.shape[1] == 384 and new_frame.image.shape[2] == 384
    assert new_frame.disparity.data.shape[2] == 384 and new_frame.disparity.data.shape[2] == 384
    assert new_frame.planar_depth.data.shape[2] == 384 and new_frame.planar_depth.data.shape[2] == 384


def test_disparity_resize():
    frame = get_frame()
    disparity = frame.disparity
    resize1 = disparity.resize((240, 320), method="interpolate")
    resize2 = disparity.resize((240, 320), method="maxpool")
    resize3 = disparity.resize((240, 320), method="minpool")
    assert resize1.data.shape[1] == 240 and resize1.data.shape[2] == 320
    assert resize2.data.shape[1] == 240 and resize2.data.shape[2] == 320
    assert resize3.data.shape[1] == 240 and resize3.data.shape[2] == 320


def test_depth_resize():
    frame = get_frame()
    planar_depth = frame.planar_depth
    resize1 = planar_depth.resize((240, 320), method="interpolate")
    resize2 = planar_depth.resize((240, 320), method="maxpool")
    resize3 = planar_depth.resize((240, 320), method="minpool")
    assert resize1.data.shape[1] == 240 and resize1.data.shape[2] == 320
    assert resize2.data.shape[1] == 240 and resize2.data.shape[2] == 320
    assert resize3.data.shape[1] == 240 and resize3.data.shape[2] == 320


def test_disparity_get_view():
    frame = get_frame()
    disparity = frame.disparity
    view = disparity.get_view()
    assert isinstance(view, np.ndarray)


def test_batched_disparity_get_view():
    frame = get_frame()
    disp2 = torch.stack([frame.disparity, frame.disparity])
    view = disp2.get_view()
    assert isinstance(view, list) and len(view) == 2


def test_depth_get_view():
    frame = get_frame()
    planar_depth = frame.planar_depth
    view = planar_depth.get_view()
    assert isinstance(view, np.ndarray)


def test_batched_depth_get_view():
    frame = get_frame()
    depth = torch.stack([frame.planar_depth, frame.planar_depth])
    view = depth.get_view()
    assert isinstance(view, list) and len(view) == 2


if __name__ == "__main__":
    test_frame_batching()
    test_frame_resize()
    test_disparity_resize()
    test_depth_resize()
    test_disparity_get_view()
    test_batched_disparity_get_view()
    test_depth_get_view()
    test_batched_depth_get_view()
