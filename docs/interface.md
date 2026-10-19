# Interface reference

Message types come from `rammp_perception_interfaces` in
[rammp-interfaces-ros2](https://github.com/rammp-org/rammp-interfaces-ros2). Every detector publishes one
detection message, `ObjectDetections`: the objects in a camera frame plus one mono16 label image of their masks.
Every node runs in the `/perception` namespace.

## Scene segmenter

Every alternative (`yoloe` on GPU or DLA, `rfdetr`) runs the same node from `rammp_detection_common`, as
`/perception/scene_segmenter`. Detections go to `/perception/object_detections`, not under the node's name, so
consumers do not depend on which segmenter is running.

### Topics

Published:

| Topic                                            | Type                                           | Notes                                                                                                                                 |
| ------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `/perception/object_detections`                  | `rammp_perception_interfaces/ObjectDetections` | every processed frame; `label_image` is mono16, pixel value = `DetectedObject.mask_id`, 0 for none. The Orchestrator reads this topic |
| `/perception/scene_segmenter/overlay/compressed` | `sensor_msgs/CompressedImage`                  | masks and labels drawn on the frame; only while subscribed and `publish_overlay` is true                                              |

Subscribed:

| Topic                                           | Type                | Notes                                               |
| ----------------------------------------------- | ------------------- | --------------------------------------------------- |
| `image_topic` (`/scene_camera/color/image_raw`) | `sensor_msgs/Image` | `rgb8`, `bgr8`, `rgba8` or `bgra8`; sensor-data QoS |

Every header is the camera frame's header, so `stamp` and `frame_id` match the image the result came from.
Ids are tracked: an object keeps its `instance_id` (`<class>_<n>`, e.g. `microwave_2`) and `mask_id` from frame to frame while the backend's tracker holds it.
YOLOE tracks with Ultralytics' built-in BoT-SORT (`model.track`), RF-DETR with Roboflow's `trackers` ByteTrack, the
pairing Roboflow recommends. A new object gets an id once it has been seen on consecutive frames; an object lost
for more than `track_buffer_frames` processed frames is dropped and gets a new id if it comes back. When two prompts or COCO classes of
one label fire on one object, it keeps the older id. `state` is always `STATE_VISIBLE`, and `has_pose` false: [BoxerNet](#boxernet-3d-boxes) adds 3D boxes on its own topic. Ids run 1 to
65535 and wrap. Where masks overlap, the higher-scoring instance owns the pixel.

### Parameters

Common to every alternative:

| Parameter             | Default                              | Meaning                                                                                                                                  |
| --------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `image_topic`         | `/scene_camera/color/image_raw`      | camera to segment                                                                                                                        |
| `frame_stride`        | `3`                                  | process every Nth camera frame (3 = 10 Hz from a 30 fps camera); a backend slower than that takes the newest frame as soon as it is free |
| `tracking`            | `true`                               | keep ids across frames with the backend's tracker; `false` numbers them per frame                                                        |
| `track_buffer_frames` | `10`                                 | processed frames a lost object keeps its id before it is dropped                                                                         |
| `classes_file`        | `/etc/rammp/perception/classes.yaml` | the class list; mount a file over it to change classes                                                                                   |
| `publish_overlay`     | `true`                               | draw `~/overlay/compressed` when something subscribes                                                                                    |
| `mock`                | `false`                              | no model and no camera: publish empty results at 10 Hz                                                                                   |

YOLOE (`rammp-perception/yoloe`):

| Parameter    | Default          | Meaning                                    |
| ------------ | ---------------- | ------------------------------------------ |
| `model`      | `yoloe-26s-seg`  | `yoloe-26s-seg` or `yoloe-26m-seg`         |
| `device`     | `gpu`            | `gpu`, `dla0` or `dla1`                    |
| `imgsz`      | `640`            | network input size                         |
| `engine_dir` | `/cache/engines` | TensorRT engine cache; mount a volume here |

A TensorRT engine is specific to the model, device, input size, class prompts and TensorRT version, and its file
name records all of them. A node that finds no matching engine builds one at startup: about 2 minutes for a DLA
core, a few minutes for the GPU. To build ahead of time:

```bash
docker run --rm --runtime nvidia -v rammp_perception_engines:/cache/engines \
  --entrypoint /ros_entrypoint.sh ghcr.io/rammp-org/rammp-perception/yoloe:<tag> \
  ros2 run rammp_yoloe build_engine --model yoloe-26s-seg --device dla0
```

RF-DETR (`rammp-perception/rfdetr`):

| Parameter | Default | Meaning                                            |
| --------- | ------- | -------------------------------------------------- |
| `model`   | `small` | COCO RF-DETR-Seg size: `nano`, `small` or `medium` |

## The class list

`object_detection/config/classes.yaml`, built into every image at `/etc/rammp/perception/classes.yaml`:

| Label          | YOLOE prompts                                                       | RF-DETR COCO classes | Threshold |
| -------------- | ------------------------------------------------------------------- | -------------------- | --------- |
| food container | bowl, food container, plastic food container, tupperware, lunch box | bowl                 | 0.25      |
| drink cup      | cup, mug, drink cup, glass                                          | cup, wine glass      | 0.25      |
| fridge         | refrigerator, fridge                                                | refrigerator         | 0.25      |
| microwave      | microwave                                                           | microwave            | 0.25      |
| table          | table, dining table                                                 | dining table         | 0.15      |

Several prompts can fire on one object. Within a label, a detection whose box overlaps a higher-scoring one by
IoU 0.7 or more is dropped, so each object is reported once. A closed, lidded container has no COCO class, so
RF-DETR finds food containers only when they look like bowls.

## SAM3

`rammp-perception/sam3`, node `/perception/sam3`. Parameters and defaults are in `object_detection/sam3/sam3.yaml`.

| Service / topic                      | Type               | What it does                                                                |
| ------------------------------------ | ------------------ | --------------------------------------------------------------------------- |
| `/perception/sam3/detect`            | `Detect`           | One high-quality detection on the next frame.                               |
| `/perception/sam3/start_tracking`    | `StartTracking`    | Detect, then track every instance and stream it. Ends any open session.     |
| `/perception/sam3/renew_tracking`    | `RenewTracking`    | Extend the open session's lease (no-op while leases are disabled).          |
| `/perception/sam3/stop_tracking`     | `StopTracking`     | End the session.                                                            |
| `/perception/sam3/object_detections` | `ObjectDetections` | Tracked frames: objects plus a mono16 label image, pixel value = `mask_id`. |
| `/perception/sam3/status`            | `TrackingStatus`   | Session state at 2 Hz.                                                      |

A tracking session runs until `/perception/sam3/stop_tracking`, or until a new `/perception/sam3/start_tracking` replaces it. Leases are
off by default (`lease_enabled: false`): `lease_s` is ignored and `/perception/sam3/renew_tracking` is a no-op. With
`lease_enabled: true`, a session ends `lease_s` after the last start or renew, so a client that crashes cannot leave
the GPU busy.

Detection runs the full model at 1008 px. Tracking runs at 336 px, seeded with the detector's masks. Measured
through ROS on abra with TensorRT encoders and a 1280x720 camera:

| Operation                               | Time                              |
| --------------------------------------- | --------------------------------- |
| Detect, 1 prompt                        | ~420-470 ms                       |
| Detect, 4 prompts                       | ~830 ms                           |
| Detect, 10 prompts                      | ~870 ms                           |
| Track 1 object                          | 12.5 Hz, ~100 ms frame-to-publish |
| Track 7 objects                         | 8.3 Hz, ~145 ms frame-to-publish  |
| Startup (weights cached, engines built) | ~12 s                             |

The image carries the weights of the gated `facebook/sam3` repo at `/models/sam3`, so it stays private (the SAM
License limits redistribution); a snapshot mounted there takes precedence. Building it needs a Hugging Face token
from an account that accepted the license: `HF_TOKEN=hf_... make sam3`, and the `HF_TOKEN` secret in CI. The
TensorRT engines are not in the image: build them once per machine into a volume (about 8 minutes on an AGX Orin;
engines already current are skipped):

```bash
docker run --rm --runtime nvidia -v sam3_engines:/cache/engines \
  --entrypoint /ros_entrypoint.sh ghcr.io/rammp-org/rammp-perception/sam3:<tag> \
  ros2 run rammp_sam3 build_engines
```

Without engines the node falls back to PyTorch and logs why.

## BoxerNet 3D boxes

`rammp-perception/boxer`, node `/perception/boxer`. It lifts every object a detector finds to a 3D box with
[BoxerNet](https://github.com/facebookresearch/boxer) and republishes the detector's `ObjectDetections` on
`/perception/object_detections_3d`, unchanged except that each visible object gets `has_pose`, `pose` and `size`. It runs
apart from the detectors, so it works with any of them (SAM3 included) and never slows one down.

`pose` is the 3D box's center, and its axes as orientation, in the message's `header.frame_id` (the camera's optical
frame); `size` is the box's full extent along those axes in meters. The box is gravity-aligned: its z axis is up in
`gravity_frame`, and BoxerNet estimates only its yaw about that axis. Yaw is ambiguous by 180 degrees: a box has no front.
`has_pose` is false for objects BoxerNet is less than `min_prob` confident about, and for every object in a frame it
could not lift (no depth, camera info or TF for that stamp; the reason is logged).

Per frame it needs the color frame the detections came from (matched by stamp), depth registered to that color frame
(Orbbec: `depth_registration: true`), the color camera's intrinsics, and TF from `gravity_frame` to the camera at the
frame's stamp. Only that transform's rotation is used, for gravity.

### Topics

| Topic                              | Type                                           | Notes                                                          |
| ---------------------------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| `/perception/object_detections_3d` | `rammp_perception_interfaces/ObjectDetections` | published: the input message with 3D boxes, every frame        |
| `detections_topic`                 | `rammp_perception_interfaces/ObjectDetections` | subscribed                                                     |
| `image_topic`, `depth_topic`       | `sensor_msgs/Image`                            | subscribed, sensor-data QoS; depth `16UC1` (mm) or `32FC1` (m) |
| `camera_info_topic`                | `sensor_msgs/CameraInfo`                       | subscribed; the color camera's                                 |

### Parameters

| Parameter           | Default                            | Meaning                                                         |
| ------------------- | ---------------------------------- | --------------------------------------------------------------- |
| `detections_topic`  | `/perception/object_detections`    | the detector to lift                                            |
| `image_topic`       | `/scene_camera/color/image_raw`    | the color frames those detections came from                     |
| `depth_topic`       | `/scene_camera/depth/image_raw`    | depth registered to the color frames                            |
| `camera_info_topic` | `/scene_camera/color/camera_info`  | the color camera's intrinsics                                   |
| `gravity_frame`     | `base_link`                        | any frame with +z up                                            |
| `min_prob`          | `0.5`                              | BoxerNet confidence below which an object gets no pose          |
| `depth_slop_s`      | `0.02`                             | how far apart in time a depth frame may be from its color frame |
| `output_topic`      | `/perception/object_detections_3d` | where to publish                                                |
| `engine_dir`        | `/cache/engines`                   | TensorRT engine cache; mount a volume here                      |

The four input topics can change at run time, to lift another detector or camera without a restart:

```bash
ros2 param set /perception/boxer detections_topic /perception/sam3/object_detections
```

Measured on abra (AGX Orin, MAXN) with the 1280x720 scene camera and YOLOE-26s on DLA0: the node takes about 90 ms
per frame (73 ms of it lifting) for any number of objects, keeping up with the detector at 8.5 Hz, and adds about
100 ms to camera-to-result latency;
GPU+SoC rail about 26 W while lifting continuously. Running a detector on the GPU at the same time slows both; on a DLA
core it does not. The image carries Boxer's code and checkpoints (CC-BY-NC 4.0, non-commercial) but not the TensorRT
engines: the first start exports and builds them into `engine_dir` (a few minutes). To build ahead of time:

```bash
docker run --rm --runtime nvidia -v rammp_perception_engines:/cache/engines \
  --entrypoint /ros_entrypoint.sh ghcr.io/rammp-org/rammp-perception/boxer:<tag> \
  ros2 run rammp_boxer build_engines
```
