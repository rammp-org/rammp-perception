from setuptools import setup

package_name = "rammp_sam3"

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
    description="SAM3 open-vocabulary detection and tracking for RAMMP",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sam3_node = rammp_sam3.node:main",
            "build_engines = rammp_sam3.build_engines:main",
        ],
    },
)
