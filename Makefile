# Perception images. Built natively on the arm64 Jetson that runs them (abra); CI publishes the same images.
#   make build                 every image, tagged :dev
#   make yoloe                 just one
#   make yoloe INTERFACES=../rammp-interfaces-ros2   build against a local interfaces checkout
REGISTRY ?= ghcr.io/rammp-org/rammp-perception
TAG ?= dev
IMAGES := yoloe rfdetr sam3 boxer
# Images whose node has a hardware-free mock mode for `make smoke`.
SMOKE_IMAGES := yoloe rfdetr

.PHONY: build push check test smoke lint stage-interfaces $(IMAGES)

build: $(IMAGES)

$(IMAGES): stage-interfaces
	docker build -f object_detection/$@/Dockerfile -t $(REGISTRY)/$@:$(TAG) .

# Images build against the tagged rammp-interfaces-ros2 release unless INTERFACES names a checkout to stage.
stage-interfaces:
	rm -rf interfaces/rammp_perception_interfaces
	$(if $(INTERFACES),cp -r $(INTERFACES)/rammp_perception_interfaces interfaces/)

check:
	uv run --with pyyaml python3 scripts/validate_fragment.py rammp-alternative*.yaml

test:
	uv run --with pyyaml --with pytest python -m pytest tests -q
	cd object_detection/common/rammp_detection_common && uv run --with pyyaml --with numpy --with pytest python -m pytest test -q
	cd object_detection/sam3/rammp_sam3 && uv run --with numpy --with pytest python -m pytest test -q
	cd object_detection/boxer/rammp_boxer && uv run --with numpy --with pytest python -m pytest test -q

smoke:
	for i in $(SMOKE_IMAGES); do ./scripts/smoke.sh $(REGISTRY)/$$i:$(TAG) || exit 1; done

lint:
	pre-commit run --all-files

push: build
	for i in $(IMAGES); do docker push $(REGISTRY)/$$i:$(TAG); done
