from setuptools import setup

package_name = "rammp_boxer"

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
    description="BoxerNet 3D boxes for any detector's ObjectDetections, from aligned depth and TF",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "boxer_node = rammp_boxer.node:main",
            "build_engines = rammp_boxer.engine:main",
        ],
    },
)
