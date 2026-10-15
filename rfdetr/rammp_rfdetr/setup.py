from setuptools import setup

package_name = "rammp_rfdetr"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Swapnil Pande",
    maintainer_email="swapnil.pande98@gmail.com",
    description="RF-DETR-Seg (COCO) instance segmentation on the GPU, publishing the shared class list through rammp_detection_common",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rfdetr_node = rammp_rfdetr.node:main",
        ],
    },
)
