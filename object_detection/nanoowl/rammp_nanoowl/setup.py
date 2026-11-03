from setuptools import setup

package_name = "rammp_nanoowl"

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
    description="NanoOWL (OWL-ViT, TensorRT FP16) open-vocabulary detection on the GPU, publishing the shared class list through rammp_detection_common",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "nanoowl_node = rammp_nanoowl.node:main",
            "build_engine = rammp_nanoowl.owl:main",
        ],
    },
)
