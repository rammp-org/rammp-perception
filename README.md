# rammp-perception

Object segmentation for RAMMP, as ROS 2 nodes in Docker images run by sheppy:

| Image                                       | Role                                          | Runs on                                |
| ------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| `ghcr.io/rammp-org/rammp-perception/yoloe`  | always-on scene segmenter, YOLOE              | GPU, or a DLA core with `device: dla0` |
| `ghcr.io/rammp-org/rammp-perception/rfdetr` | always-on scene segmenter, RF-DETR-Seg (COCO) | GPU                                    |
| `ghcr.io/rammp-org/rammp-perception/sam3`   | on-demand expert detection and tracking, SAM3 | GPU                                    |

Every detector publishes one detection message, `rammp_perception_interfaces/ObjectDetections`: the objects
(`instance_id` like `microwave_2`, stable while tracked, class, score, box) plus one mono16 label image of
their masks. The three scene segmenter alternatives (YOLOE on the GPU, YOLOE on the DLA, RF-DETR) publish it on
the same topic with the same class names, so a sheppy profile swaps between them:

| Topic                                            | Type                                           | Publisher                                                 |
| ------------------------------------------------ | ---------------------------------------------- | --------------------------------------------------------- |
| `/perception/object_detections`                  | `rammp_perception_interfaces/ObjectDetections` | the scene segmenter, always on; the Orchestrator reads it |
| `/perception/sam3/object_detections`             | `rammp_perception_interfaces/ObjectDetections` | SAM3, during a tracking session                           |
| `/perception/scene_segmenter/overlay/compressed` | `sensor_msgs/CompressedImage`                  | masks drawn on the frame, for viewing                     |

The class list (food container, drink cup, fridge, microwave, table) and the prompts and COCO classes behind each
label are in [`object_detection/config/classes.yaml`](object_detection/config/classes.yaml).

Full docs: [`docs/`](docs/index.mdx), published on the RAMMP docs site. Topics, parameters and the class list are in
the [interface reference](docs/interface.md).

## Layout

```
object_detection/
  config/classes.yaml            the shared class list, built into every image
  common/rammp_detection_common  class list, duplicate merging, and the node every segmenter runs in
  yoloe/                         rammp_yoloe package and Dockerfile
  rfdetr/                        rammp_rfdetr package and Dockerfile
  sam3/                          rammp_sam3 package, Dockerfile and default parameters
interfaces/                      staging for an unreleased rammp_perception_interfaces (gitignored)
rammp-alternative.*.yaml         sheppy alternatives for rammp-deployments
```

## Build and test

On an arm64 Jetson (abra):

```bash
make build                                       # every image, :dev
make yoloe INTERFACES=../rammp-interfaces-ros2   # against an unreleased interfaces checkout
make test                                        # unit tests, no ROS or GPU
make check                                       # validate the sheppy fragments
make smoke                                       # each segmenter image publishes in mock mode and exits on SIGTERM
```

`rammp_perception_interfaces` comes from rammp-interfaces-ros2 at `INTERFACES_REF` (default `v1.2.0`) until a
`rammp-cuda` release carries it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and PR workflow.
