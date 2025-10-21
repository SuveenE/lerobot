# Depth Data Collection Guide

This guide explains how to collect depth data alongside RGB images in the LeRobot framework.

## Overview

The LeRobot framework now supports optional depth data collection from depth-capable cameras. Depth data is stored as separate 2D arrays (uint16 format, millimeters) alongside RGB images.

## Key Features

- **Optional by design**: Depth collection is disabled by default and can be enabled per camera
- **Separate streams**: Depth data is captured as separate 2D arrays, not as a 4th channel
- **Standard format**: Depth stored as uint16 (raw depth in millimeters, typically 0-10000mm range)
- **Backward compatible**: Existing recordings without depth continue to work unchanged
- **Async support**: Depth captured asynchronously like RGB for optimal performance

## Supported Cameras

### RealSense Cameras (Full Support)

- **Intel RealSense D405, D435, D455** and other depth-capable models
- RGB and depth streams both supported through `intelrealsense` camera type
- Enable depth with `use_depth=True` configuration flag

### Orbbec Gemini Cameras (Partial Support)

- **Orbbec Gemini 336L, Gemini E** and other models
- **RGB**: Use OpenCV camera type (UVC support, no special drivers needed)
- **Depth**: Requires Orbbec SDK implementation (not yet implemented)

## Configuration

### Basic Configuration (No Depth)

```python
from lerobot.robots.bi_piper.config_bi_piper import BiPiperConfig

config = BiPiperConfig(
    type="bi_piper",
    left_arm_can_port="can_0",
    right_arm_can_port="can_1",
    cameras={
        "front": {
            "type": "opencv",
            "index_or_path": 0,
            "width": 640,
            "height": 480,
            "fps": 30
        },
    }
)
```

### Configuration with Depth Enabled

```python
from lerobot.robots.bi_piper.config_bi_piper import BiPiperConfig

config = BiPiperConfig(
    type="bi_piper",
    left_arm_can_port="can_0",
    right_arm_can_port="can_1",
    cameras={
        "front": {
            "type": "intelrealsense",
            "serial_number_or_name": "123456789",  # Your camera's serial number
            "width": 640,
            "height": 480,
            "fps": 30,
            "use_depth": True  # Enable depth capture
        },
        "left": {
            "type": "opencv",  # Orbbec RGB via OpenCV
            "index_or_path": 1,
            "width": 640,
            "height": 480,
            "fps": 30
        }
    }
)
```

## Command Line Usage

### Recording with Depth Data

```bash
python -m lerobot.record \
    --robot.type=bi_piper \
    --robot.left_arm_can_port=can_0 \
    --robot.right_arm_can_port=can_1 \
    --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '123456789', width: 640, height: 480, fps: 30, use_depth: true}}" \
    --dataset.repo_id=your_username/dataset-with-depth \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick and place" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10
```

## Data Format

### Dataset Structure

When depth is enabled for a camera named `front`, the dataset will contain:

- **RGB data**: `observation.images.front` - Shape: (height, width, 3), dtype: uint8
- **Depth data**: `observation.images.front_depth` - Shape: (height, width), dtype: uint16

### Depth Values

- **Format**: uint16 (unsigned 16-bit integer)
- **Units**: Millimeters
- **Range**: Typically 0-10000mm (0-10 meters)
- **Special values**:
  - `0`: Invalid/no measurement
  - Valid measurements typically start from ~100mm depending on camera

### Example: Accessing Depth Data

```python
from lerobot.datasets import LeRobotDataset

# Load dataset with depth
dataset = LeRobotDataset("your_username/dataset-with-depth")

# Access first frame
frame = dataset[0]

# RGB image
rgb_image = frame["observation.images.front"]  # Shape: (H, W, 3), dtype: uint8

# Depth image
depth_image = frame["observation.images.front_depth"]  # Shape: (H, W), dtype: uint16

# Convert depth to meters
depth_meters = depth_image.astype(float) / 1000.0

# Visualize
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(rgb_image)
ax1.set_title("RGB Image")
ax2.imshow(depth_meters, cmap='viridis')
ax2.set_title("Depth Map (meters)")
plt.show()
```

## Implementation Details

### Robot Configuration

Robots need to update two methods to support depth:

1. **`_cameras_ft` property**: Add depth features for cameras with `use_depth=True`

```python
@property
def _cameras_ft(self) -> dict[str, tuple]:
    features = {}
    for cam_name in self.cameras:
        cfg = self.config.cameras[cam_name]
        # RGB features
        features[cam_name] = (cfg.height, cfg.width, 3)
        # Depth features if enabled
        if hasattr(cfg, 'use_depth') and cfg.use_depth:
            features[f"{cam_name}_depth"] = (cfg.height, cfg.width)
    return features
```

2. **`get_observation` method**: Capture depth alongside RGB

```python
def get_observation(self) -> dict:
    observation = {}
    # ... other observations ...

    # Capture camera images
    for cam_name, cam in self.cameras.items():
        observation[cam_name] = cam.async_read()
        # Capture depth if enabled
        cfg = self.config.cameras[cam_name]
        if hasattr(cfg, 'use_depth') and cfg.use_depth:
            if hasattr(cam, 'async_read_depth'):
                observation[f"{cam_name}_depth"] = cam.async_read_depth()

    return observation
```

### Camera Implementation

Cameras supporting depth need to implement:

1. **`use_depth` configuration flag**: Boolean to enable/disable depth capture
2. **`read_depth()` method**: Synchronous depth frame capture
3. **`async_read_depth()` method**: Asynchronous depth frame capture (recommended)

See `src/lerobot/cameras/realsense/camera_realsense.py` for reference implementation.

## Finding Camera Serial Numbers

### RealSense Cameras

```bash
# Find all connected RealSense cameras
python -m lerobot.find_cameras realsense
```

This will display:

- Serial numbers
- Camera model names
- Firmware versions
- Supported resolutions and frame rates

### OpenCV Cameras (for Orbbec RGB)

```bash
# Find all OpenCV-compatible cameras
python -m lerobot.find_cameras opencv
```

## Troubleshooting

### Depth Capture Fails

1. **Check if depth is enabled**: Verify `use_depth=True` in camera config
2. **Verify camera support**: Ensure your camera model supports depth (RealSense D-series)
3. **Check USB connection**: Depth requires USB 3.0 for most cameras
4. **Update firmware**: Use RealSense Viewer or Orbbec Viewer to update camera firmware

### Depth Resolution Mismatch

- Depth resolution must match RGB resolution in most cases
- Check camera datasheet for supported depth resolutions
- Some cameras support different resolutions for RGB and depth

### Performance Issues

- Depth capture adds overhead (~2x frame processing time)
- Use async capture (`async_read_depth()`) for better performance
- Consider lower frame rates if performance is critical (e.g., 15 fps instead of 30 fps)
- Reduce resolution if needed (e.g., 640x480 instead of 1280x720)

## Requirements

### Software

```bash
# For RealSense cameras
pip install pyrealsense2

# For Orbbec cameras (RGB via OpenCV)
# No additional packages needed, uses standard OpenCV

# For Orbbec depth (not yet implemented)
# pip install pyorbbecsdk  # When Orbbec driver is implemented
```

### Hardware

- **RealSense D405/D435**: USB 3.0 or higher
- **Orbbec Gemini 336L/E**: USB 3.0 Type-C
- Sufficient USB bandwidth (use dedicated USB controller if multiple cameras)

## Future Work

### Orbbec Depth Support

To add Orbbec depth support, implement:

1. Create `src/lerobot/cameras/orbbec/` directory
2. Implement `OrbbecCamera` class using `pyorbbecsdk`
3. Add `read_depth()` and `async_read_depth()` methods
4. Register "orbbec" camera type in `src/lerobot/cameras/utils.py`

See `src/lerobot/cameras/realsense/` for reference implementation.

## Additional Resources

- [Intel RealSense Documentation](https://dev.intelrealsense.com/)
- [Orbbec SDK Documentation](https://www.orbbec.com/developers/)
- [LeRobot Camera Documentation](./cameras.mdx)
- [Example: BiPiper with Depth](../examples/bi_piper_example.py)
