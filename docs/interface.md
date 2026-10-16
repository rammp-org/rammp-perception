# Interface reference

Message types come from `rammp_perception_interfaces` in
[rammp-interfaces-ros2](https://github.com/rammp-org/rammp-interfaces-ros2) and from the standard `vision_msgs`.

## Scene segmenter

Every alternative (`yoloe` on GPU or DLA, `rfdetr`) runs the same node from `rammp_detection_common`. Its default
name is `scene_segmenter`.

### Topics

Published:

| Topic                                 | Type                                        | Notes                                                                                    |
| ------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `/scene_segmenter/instances`          | `rammp_perception_interfaces/InstanceMasks` | every processed frame; `label_image` is mono16, pixel value = `Instance.id`, 0 for none  |
| `/scene_segmenter/detections`         | `vision_msgs/Detection2DArray`              | the same instances as boxes; `Detection2D.id` = `Instance.id`, `class_id` = label        |
| `/scene_segmenter/overlay/compressed` | `sensor_msgs/CompressedImage`               | masks and labels drawn on the frame; only while subscribed and `publish_overlay` is true |

Subscribed:

| Topic                                           | Type                | Notes                                               |
| ----------------------------------------------- | ------------------- | --------------------------------------------------- |
| `image_topic` (`/scene_camera/color/image_raw`) | `sensor_msgs/Image` | `rgb8`, `bgr8`, `rgba8` or `bgra8`; sensor-data QoS |

Every header is the camera frame's header, so `stamp` and `frame_id` match the image the result came from.
Ids are tracked: an object keeps its `Instance.id` from frame to frame while the backend's tracker holds it.
YOLOE tracks with Ultralytics' built-in BoT-SORT (`model.track`), RF-DETR with Roboflow's `trackers` ByteTrack, the
pairing Roboflow recommends. A new object gets an id once it has been seen on consecutive frames; an object lost
for more than `track_buffer_frames` processed frames is dropped and gets a new id if it comes back. When two prompts or COCO classes of
one label fire on one object, it keeps the older id. `Instance.state` is always `STATE_VISIBLE`. Ids run 1 to
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

`rammp-perception/sam3`, node `sam3`. Parameters and defaults are in `object_detection/sam3/sam3.yaml`.

| Service / topic        | Type             | What it does                                                                    |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------- |
| `/sam3/detect`         | `Detect`         | One high-quality detection on the next frame.                                   |
| `/sam3/start_tracking` | `StartTracking`  | Detect, then track every instance and stream it. Ends any open session.         |
| `/sam3/renew_tracking` | `RenewTracking`  | Extend the open session's lease (no-op while leases are disabled).              |
| `/sam3/stop_tracking`  | `StopTracking`   | End the session.                                                                |
| `/sam3/instances`      | `InstanceMasks`  | Tracked frames: instances plus a mono16 label image, pixel value = instance id. |
| `/sam3/status`         | `TrackingStatus` | Session state at 2 Hz.                                                          |

A tracking session runs until `/sam3/stop_tracking`, or until a new `/sam3/start_tracking` replaces it. Leases are
off by default (`lease_enabled: false`): `lease_s` is ignored and `/sam3/renew_tracking` is a no-op. With
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

The image carries neither the weights nor the TensorRT engines. Mount a Hugging Face snapshot of the gated
`facebook/sam3` repo at `/models/sam3`, and build the engines once per machine into a volume (about 8 minutes on
an AGX Orin; engines already current are skipped):

```bash
docker run --rm --runtime nvidia -v /path/to/sam3:/models/sam3:ro -v sam3_engines:/cache/engines \
  --entrypoint /ros_entrypoint.sh ghcr.io/rammp-org/rammp-perception/sam3:<tag> \
  ros2 run rammp_sam3 build_engines
```

Without engines the node falls back to PyTorch and logs why.
