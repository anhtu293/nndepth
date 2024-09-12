# Scene
## 1. Introduction
Core module for scene representation and manipulation:
### Frame: Comprehensive representation of an image frame, including:
- Raw image data
- Camera intrinsics and extrinsics
- Associated depth (I name it `planar_depth` in `Frame` object since `depth` keyword is reserved for `torch.Tensor`) and disparity maps
- Pose information
### Depth: Robust depth map handling with features for:
- Efficient resizing and interpolation
- Customizable visualization options
- Conversion to other formats (e.g., point clouds)
### Disparity: Specialized class for stereo vision, offering:
- Conversion between disparity and depth
- Stereo-specific visualization tools
- Compatibility with various stereo algorithms

## 2. Some exmaples of **scene**

### 2.1 Create scene object
```Python
>>> import torch
>>> from nndepth.scene import Frame, Disparity, Depth
>>> image = torch.ones((3, 480, 640))  # image data
>>>
>>> depth = torch.ones((1, image.shape[-2], image.shape[-1]))  # Depth data
>>> depth = Depth(data=depth) # Depth object
>>>
>>> disparity = torch.ones((1, image.shape[-2], image.shape[-1]))  # Disparity data
>>> disparity = Disparity(data=disparity, disp_sign="negative")  # Disparity object
>>>
>>> cam_intrinsic = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1)  # Intrinsic matrix, in torch.Tensor
>>> cam_ext = torch.eye(4)  # Camera extrinsic matrix
>>> pose = torch.eye(4)  # Pose matrix
>>>
>>> frame = Frame(
    image=torch.Tensor(image),
    disparity=disparity,
    planar_depth=depth,
    cam_intrinsic=cam_intrinsic,
    cam_extrinsic=cam_ext,
    pose=pose
)  # Create Frame object with all information. It is possible to create Frame object with only `image` data, other parameters have None as default value.
>>> frame
Frame(
    cam_extrinsic=Tensor(shape=torch.Size([4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    cam_intrinsic=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    disparity=Disparity(
        data=Tensor(shape=torch.Size([1, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
        disp_sign=NonTensorData(data=negative, batch_size=torch.Size([]), device=None),
        occlusion=None,
        batch_size=torch.Size([]),
        device=None,
        is_shared=False),
    image=Tensor(shape=torch.Size([3, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
    planar_depth=Depth(
        data=Tensor(shape=torch.Size([1, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
        valid_mask=None,
        batch_size=torch.Size([]),
        device=None,
        is_shared=False),
    pose=Tensor(shape=torch.Size([4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    baseline=None,
    camera=None,
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)  # batch_size is empty here since there is no batch dimension
```

### 2.2 Batching
```python
>>> batch_1 = frame.unsqueeze(0)
>>> batch_1
Frame(
    cam_extrinsic=Tensor(shape=torch.Size([1, 4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    cam_intrinsic=Tensor(shape=torch.Size([1, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    disparity=Disparity(
        data=Tensor(shape=torch.Size([1, 1, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
        disp_sign=NonTensorData(data=negative, batch_size=torch.Size([1]), device=None),
        occlusion=None,
        batch_size=torch.Size([1]),
        device=None,
        is_shared=False),
    image=Tensor(shape=torch.Size([1, 3, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
    planar_depth=Depth(
        data=Tensor(shape=torch.Size([1, 1, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
        valid_mask=None,
        batch_size=torch.Size([1]),
        device=None,
        is_shared=False),
    pose=Tensor(shape=torch.Size([1, 4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    baseline=None,
    camera=None,
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)  # batch_size is 1 here
>>>
>>> from copy import deepcopy
>>> frame2 = deepcopy(frame)
>>> # We create a batch of 2 frames by concatenating 2 frames here
>>> batch = torch.stack([frame, frame2])
>>> batch
Frame(
    cam_extrinsic=Tensor(shape=torch.Size([2, 4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    cam_intrinsic=Tensor(shape=torch.Size([2, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    disparity=Disparity(
        data=Tensor(shape=torch.Size([2, 1, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
        disp_sign=NonTensorData(data=negative, batch_size=torch.Size([2]), device=None),
        occlusion=None,
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False),
    image=Tensor(shape=torch.Size([2, 3, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
    planar_depth=Depth(
        data=Tensor(shape=torch.Size([2, 1, 480, 640]), device=cpu, dtype=torch.float32, is_shared=False),
        valid_mask=None,
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False),
    pose=Tensor(shape=torch.Size([2, 4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    baseline=None,
    camera=None,
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)  # batch_size is change to torch.Size([2]). Notice that all the labels (planar_depth, disparity, intrinsic, extrinsic, pose) are batched automatically.
```

### 2.3 Resize
```python
>>> resized = batch.resize((384,384))
>>> resized
Frame(
    cam_extrinsic=Tensor(shape=torch.Size([2, 4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    cam_intrinsic=Tensor(shape=torch.Size([2, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    disparity=Disparity(
        data=Tensor(shape=torch.Size([2, 1, 384, 384]), device=cpu, dtype=torch.float32, is_shared=False),
        disp_sign=NonTensorData(data=negative, batch_size=torch.Size([2]), device=None),
        occlusion=None,
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False),
    image=Tensor(shape=torch.Size([2, 3, 384, 384]), device=cpu, dtype=torch.float32, is_shared=False),
    planar_depth=Depth(
        data=Tensor(shape=torch.Size([2, 1, 384, 384]), device=cpu, dtype=torch.float32, is_shared=False),
        valid_mask=None,
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False),
    pose=None,
    baseline=None,
    camera=None,
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)
>>>  # All spatial labels (depth, disparity) are resized.
>>> # Note that Depth or Disparity can be resized seperately
>>> frame.disparity.resize((100, 100))
Disparity(
    data=Tensor(shape=torch.Size([1, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
    disp_sign=NonTensorData(data=negative, batch_size=torch.Size([]), device=None),
    occlusion=None,
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
>>> # For depth and disparity, there are 3 methods to resize: by interpolation (by default), maxpooling and minpooling.
>>> # The resizing by interpolation creates artifacts and those artifacts are usually problems for real application since they are noise.
>>> # Therefore, resizing by maxpool and minpool is prefered in some cases.
>>> frame.disparity.resize((100, 100), method="maxpool")
Disparity(
    data=Tensor(shape=torch.Size([1, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
    disp_sign=NonTensorData(data=negative, batch_size=torch.Size([]), device=None),
    occlusion=None,
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
>>> frame.disparity.resize((100, 100), method="minpool")
Disparity(
    data=Tensor(shape=torch.Size([1, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
    disp_sign=NonTensorData(data=negative, batch_size=torch.Size([]), device=None),
    occlusion=None,
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
>>>
>>> frame.planar_depth.resize((100, 100), method="maxpool")
Depth(
    data=Tensor(shape=torch.Size([1, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
    valid_mask=None,
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
>>> frame.planar_depth.resize((100, 100), method="minpool")
Depth(
    data=Tensor(shape=torch.Size([1, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
    valid_mask=None,
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
>>> # We can choose the method to resize depth and disparity when we resize frame object
>>> frame.resize((100, 100), disparity_resize_method="minpool", depth_resize_method="maxpool")
Frame(
    cam_extrinsic=Tensor(shape=torch.Size([4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    cam_intrinsic=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
    disparity=Disparity(
        data=Tensor(shape=torch.Size([1, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
        disp_sign=NonTensorData(data=negative, batch_size=torch.Size([]), device=None),
        occlusion=None,
        batch_size=torch.Size([]),
        device=None,
        is_shared=False),
    image=Tensor(shape=torch.Size([3, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
    planar_depth=Depth(
        data=Tensor(shape=torch.Size([1, 100, 100]), device=cpu, dtype=torch.float32, is_shared=False),
        valid_mask=None,
        batch_size=torch.Size([]),
        device=None,
        is_shared=False),
    pose=None,
    baseline=None,
    camera=None,
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
```

### 2.4 Visualization
- **Disparity** and **Depth** are provided **get_view** method to create quickly visualization. The return is a `np.ndarray` when there is no batch dimension and list of `np.ndarray` in case of batch.
- You can use `cv2` or `matplotlib`to visualize that array.
```python
>>> view = frame.disparity.get_view()
>>> isinstance(view, np.ndarray)
True
>>> view2 = batch.disparity.get_view()
>>> isinstance(view2, list)
True
>>> # You can choose `min` and `max` value to normalize depth/disparity map.
>>> # Default value of `min` and `max` is None: function will use the min and max value of depth/disparity to normalize
>>> # You can choose color map. List of color maps is here: https://matplotlib.org/stable/users/explain/colors/colormaps.html
```
