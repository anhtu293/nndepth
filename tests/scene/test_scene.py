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
    assert new_frame.disparity.data.shape[1] == 384 and new_frame.disparity.data.shape[2] == 384
    assert new_frame.planar_depth.data.shape[1] == 384 and new_frame.planar_depth.data.shape[2] == 384


def test_batch_resize():
    frame = get_frame()
    frame = torch.stack([frame, frame])
    new_frame = frame.resize((384, 384))
    assert new_frame.image.shape[-2] == 384 and new_frame.image.shape[-1] == 384
    assert new_frame.disparity.data.shape[-2] == 384 and new_frame.disparity.data.shape[-1] == 384
    assert new_frame.planar_depth.data.shape[-2] == 384 and new_frame.planar_depth.data.shape[-1] == 384
    assert new_frame.batch_size[0] == 2


def test_frame_resize_w_options():
    frame = get_frame()
    new_frame1 = frame.resize(
        (384, 384),
        disparity_resize_method="maxpool",
        depth_resize_method="maxpool"
    )
    new_frame2 = frame.resize(
        (384, 384),
        disparity_resize_method="minpool",
        depth_resize_method="minpool"
    )
    assert new_frame1.image.shape[1] == 384 and new_frame1.image.shape[2] == 384
    assert new_frame1.disparity.data.shape[1] == 384 and new_frame1.disparity.data.shape[2] == 384
    assert new_frame1.planar_depth.data.shape[1] == 384 and new_frame1.planar_depth.data.shape[2] == 384
    assert new_frame2.image.shape[1] == 384 and new_frame2.image.shape[2] == 384
    assert new_frame2.disparity.data.shape[1] == 384 and new_frame2.disparity.data.shape[2] == 384
    assert new_frame2.planar_depth.data.shape[1] == 384 and new_frame2.planar_depth.data.shape[2] == 384


def test_frame_resize_w_options2():
    frame = get_frame()
    new_frame1 = frame.resize(
        (320, 320),
        disparity_resize_method="maxpool",
        depth_resize_method="maxpool"
    )
    new_frame2 = frame.resize(
        (320, 320),
        disparity_resize_method="minpool",
        depth_resize_method="minpool"
    )
    assert new_frame1.image.shape[1] == 320 and new_frame1.image.shape[2] == 320
    assert new_frame1.disparity.data.shape[1] == 320 and new_frame1.disparity.data.shape[2] == 320
    assert new_frame1.planar_depth.data.shape[1] == 320 and new_frame1.planar_depth.data.shape[2] == 320
    assert new_frame2.image.shape[1] == 320 and new_frame2.image.shape[2] == 320
    assert new_frame2.disparity.data.shape[1] == 320 and new_frame2.disparity.data.shape[2] == 320
    assert new_frame2.planar_depth.data.shape[1] == 320 and new_frame2.planar_depth.data.shape[2] == 320


def test_batch_resize_w_options():
    frame = get_frame()
    frame = torch.stack([frame, frame])
    new_frame1 = frame.resize(
        (384, 384),
        disparity_resize_method="maxpool",
        depth_resize_method="maxpool"
    )
    new_frame2 = frame.resize(
        (384, 384),
        disparity_resize_method="minpool",
        depth_resize_method="minpool"
    )
    assert new_frame1.image.shape[-2] == 384 and new_frame1.image.shape[-1] == 384
    assert new_frame1.disparity.data.shape[-2] == 384 and new_frame1.disparity.data.shape[-1] == 384
    assert new_frame1.planar_depth.data.shape[-2] == 384 and new_frame1.planar_depth.data.shape[-1] == 384
    assert new_frame1.batch_size[0] == 2
    assert new_frame2.image.shape[-2] == 384 and new_frame2.image.shape[-1] == 384
    assert new_frame2.disparity.data.shape[-2] == 384 and new_frame2.disparity.data.shape[-1] == 384
    assert new_frame2.planar_depth.data.shape[-2] == 384 and new_frame2.planar_depth.data.shape[-1] == 384
    assert new_frame2.batch_size[0] == 2


def test_batch_resize_w_options2():
    frame = get_frame()
    frame = torch.stack([frame, frame])
    new_frame1 = frame.resize(
        (320, 320),
        disparity_resize_method="maxpool",
        depth_resize_method="maxpool"
    )
    new_frame2 = frame.resize(
        (320, 320),
        disparity_resize_method="minpool",
        depth_resize_method="minpool"
    )
    assert new_frame1.image.shape[-2] == 320 and new_frame1.image.shape[-1] == 320
    assert new_frame1.disparity.data.shape[-2] == 320 and new_frame1.disparity.data.shape[-1] == 320
    assert new_frame1.planar_depth.data.shape[-2] == 320 and new_frame1.planar_depth.data.shape[-1] == 320
    assert new_frame1.batch_size[0] == 2
    assert new_frame2.image.shape[-2] == 320 and new_frame2.image.shape[-1] == 320
    assert new_frame2.disparity.data.shape[-2] == 320 and new_frame2.disparity.data.shape[-1] == 320
    assert new_frame2.planar_depth.data.shape[-2] == 320 and new_frame2.planar_depth.data.shape[-1] == 320
    assert new_frame2.batch_size[0] == 2


def test_bare_frame_resize():
    frame = get_frame()
    frame.disparity = None
    frame.planar_depth = None
    frame.cam_intrinsic = None
    frame.cam_extrinsic = None
    frame.pose = None
    new_frame = frame.resize((384, 384))
    assert new_frame.image.shape[1] == 384 and new_frame.image.shape[2] == 384


def test_bare_batch_resize():
    frame = get_frame()
    frame.disparity = None
    frame.planar_depth = None
    frame.cam_intrinsic = None
    frame.cam_extrinsic = None
    frame.pose = None
    frame = torch.stack([frame, frame])
    new_frame = frame.resize((384, 384))
    assert new_frame.image.shape[-2] == 384 and new_frame.image.shape[-1] == 384
    assert new_frame.batch_size[0] == 2


def test_disparity_resize():
    frame = get_frame()
    disparity = frame.disparity
    resize1 = disparity.resize((240, 320), method="interpolate")
    resize2 = disparity.resize((240, 320), method="maxpool")
    resize3 = disparity.resize((240, 320), method="minpool")
    assert resize1.data.shape[1] == 240 and resize1.data.shape[2] == 320
    assert resize2.data.shape[1] == 240 and resize2.data.shape[2] == 320
    assert resize3.data.shape[1] == 240 and resize3.data.shape[2] == 320


def test_batch_disparity_resize():
    frame = get_frame()
    disparity = frame.disparity
    disparity = torch.stack([disparity, disparity])
    resize1 = disparity.resize((240, 320), method="interpolate")
    resize2 = disparity.resize((240, 320), method="maxpool")
    resize3 = disparity.resize((240, 320), method="minpool")
    assert resize1.data.shape[-2] == 240 and resize1.data.shape[-1] == 320
    assert resize2.data.shape[-2] == 240 and resize2.data.shape[-1] == 320
    assert resize3.data.shape[-2] == 240 and resize3.data.shape[-1] == 320


def test_depth_resize():
    frame = get_frame()
    planar_depth = frame.planar_depth
    resize1 = planar_depth.resize((240, 320), method="interpolate")
    resize2 = planar_depth.resize((240, 320), method="maxpool")
    resize3 = planar_depth.resize((240, 320), method="minpool")
    assert resize1.data.shape[1] == 240 and resize1.data.shape[2] == 320
    assert resize2.data.shape[1] == 240 and resize2.data.shape[2] == 320
    assert resize3.data.shape[1] == 240 and resize3.data.shape[2] == 320


def test_batch_depth_resize():
    frame = get_frame()
    planar_depth = frame.planar_depth
    planar_depth = torch.stack([planar_depth, planar_depth])
    resize1 = planar_depth.resize((240, 320), method="interpolate")
    resize2 = planar_depth.resize((240, 320), method="maxpool")
    resize3 = planar_depth.resize((240, 320), method="minpool")
    assert resize1.data.shape[-2] == 240 and resize1.data.shape[-1] == 320
    assert resize2.data.shape[-2] == 240 and resize2.data.shape[-1] == 320
    assert resize3.data.shape[-2] == 240 and resize3.data.shape[-1] == 320


def test_disparity_w_occ_resize():
    frame = get_frame()
    disparity = frame.disparity
    occ = torch.rand(disparity.data.shape) > 0.5
    disparity.occlusion = occ
    resize1 = disparity.resize((240, 320), method="interpolate")
    resize2 = disparity.resize((240, 320), method="maxpool")
    resize3 = disparity.resize((240, 320), method="minpool")
    assert resize1.data.shape[1] == 240 and resize1.data.shape[2] == 320
    assert resize2.data.shape[1] == 240 and resize2.data.shape[2] == 320
    assert resize3.data.shape[1] == 240 and resize3.data.shape[2] == 320
    assert resize1.occlusion.shape[1] == 240 and resize1.occlusion.shape[2] == 320
    assert resize2.occlusion.shape[1] == 240 and resize2.occlusion.shape[2] == 320
    assert resize3.occlusion.shape[1] == 240 and resize3.occlusion.shape[2] == 320


def test_depth_w_mask_resize():
    frame = get_frame()
    planar_depth = frame.planar_depth
    valid_mask = torch.rand(planar_depth.data.shape) > 0.5
    planar_depth.valid_mask = valid_mask
    resize1 = planar_depth.resize((240, 320), method="interpolate")
    resize2 = planar_depth.resize((240, 320), method="maxpool")
    resize3 = planar_depth.resize((240, 320), method="minpool")
    assert resize1.data.shape[1] == 240 and resize1.data.shape[2] == 320
    assert resize2.data.shape[1] == 240 and resize2.data.shape[2] == 320
    assert resize3.data.shape[1] == 240 and resize3.data.shape[2] == 320
    assert resize1.valid_mask.shape[1] == 240 and resize1.valid_mask.shape[2] == 320
    assert resize2.valid_mask.shape[1] == 240 and resize2.valid_mask.shape[2] == 320
    assert resize3.valid_mask.shape[1] == 240 and resize3.valid_mask.shape[2] == 320


def test_disparity_w_occ_get_view():
    frame = get_frame()
    disparity = frame.disparity
    occ = torch.rand(disparity.data.shape) > 0.5
    disparity.occlusion = occ
    view = disparity.get_view()
    assert isinstance(view, np.ndarray)


def test_batched_disparity_w_occ_get_view():
    frame = get_frame()
    disparity = frame.disparity
    occ = torch.rand(disparity.data.shape) > 0.5
    disparity.occlusion = occ
    disp2 = torch.stack([disparity, disparity])
    view = disp2.get_view()
    assert isinstance(view, list) and len(view) == 2


def test_depth_w_mask_get_view():
    frame = get_frame()
    planar_depth = frame.planar_depth
    valid_mask = torch.rand(planar_depth.data.shape) > 0.5
    planar_depth.valid_mask = valid_mask
    view = planar_depth.get_view()
    assert isinstance(view, np.ndarray)


def test_batched_depth_w_mask_get_view():
    frame = get_frame()
    planar_depth = frame.planar_depth
    valid_mask = torch.rand(planar_depth.data.shape) > 0.5
    planar_depth.valid_mask = valid_mask
    depth = torch.stack([planar_depth, planar_depth])
    view = depth.get_view()
    assert isinstance(view, list) and len(view) == 2


def test_disparity_get_view():
    frame = get_frame()
    disparity = frame.disparity
    view = disparity.get_view()
    view2 = disparity.get_view(reverse=True)
    view3 = disparity.get_view(cmap="red2green")
    assert isinstance(view, np.ndarray)
    assert isinstance(view2, np.ndarray)
    assert isinstance(view3, np.ndarray)


def test_batched_disparity_get_view():
    frame = get_frame()
    disp2 = torch.stack([frame.disparity, frame.disparity])
    view = disp2.get_view()
    view2 = disp2.get_view(reverse=True)
    view3 = disp2.get_view(cmap="red2green")
    assert isinstance(view, list) and len(view) == 2
    assert isinstance(view2, list) and len(view) == 2
    assert isinstance(view3, list) and len(view) == 2


def test_depth_get_view():
    frame = get_frame()
    planar_depth = frame.planar_depth
    view = planar_depth.get_view()
    view2 = planar_depth.get_view(reverse=True)
    view3 = planar_depth.get_view(cmap="red2green")
    assert isinstance(view, np.ndarray)
    assert isinstance(view2, np.ndarray)
    assert isinstance(view3, np.ndarray)


def test_batched_depth_get_view():
    frame = get_frame()
    depth = torch.stack([frame.planar_depth, frame.planar_depth])
    view = depth.get_view()
    view2 = depth.get_view(reverse=True)
    view3 = depth.get_view(cmap="red2green")
    assert isinstance(view, list) and len(view) == 2
    assert isinstance(view2, list) and len(view) == 2
    assert isinstance(view3, list) and len(view) == 2
