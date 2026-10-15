# rammp-perception

Object segmentation for RAMMP, as ROS 2 nodes in Docker images run by sheppy:

| Image                                       | Role                                          | Runs on                                |
| ------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| `ghcr.io/rammp-org/rammp-perception/yoloe`  | always-on scene segmenter, YOLOE              | GPU, or a DLA core with `device: dla0` |
| `ghcr.io/rammp-org/rammp-perception/rfdetr` | always-on scene segmenter, RF-DETR-Seg (COCO) | GPU                                    |
| `ghcr.io/rammp-org/rammp-perception/sam3`   | on-demand expert detection and tracking, SAM3 | GPU                                    |

The three scene segmenter alternatives (YOLOE on the GPU, YOLOE on the DLA, RF-DETR) publish the same topics,
types and class names, so a sheppy profile swaps between them:

| Topic                                 | Type                                                             |
| ------------------------------------- | ---------------------------------------------------------------- |
| `/scene_segmenter/instances`          | `rammp_perception_interfaces/InstanceMasks` (mono16 label image) |
| `/scene_segmenter/detections`         | `vision_msgs/Detection2DArray`                                   |
| `/scene_segmenter/overlay/compressed` | `sensor_msgs/CompressedImage`                                    |

The class list (food container, drink cup, fridge, microwave, table) and the prompts and COCO classes behind each
label are in [`config/classes.yaml`](config/classes.yaml).

Full docs: [`docs/`](docs/index.mdx), published on the RAMMP docs site. Topics, parameters and the class list are in
the [interface reference](docs/interface.md).

## Layout

```
config/classes.yaml          the shared class list, built into every image
common/rammp_detection_common  class list, duplicate merging, and the node every segmenter runs in
yoloe/                       rammp_yoloe package and Dockerfile
rfdetr/                      rammp_rfdetr package and Dockerfile
sam3/                        rammp_sam3 package, Dockerfile and default parameters
interfaces/                  staging for an unreleased rammp_perception_interfaces (gitignored)
rammp-alternative.*.yaml     sheppy alternatives for rammp-deployments
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
