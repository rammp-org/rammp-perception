from setuptools import setup

package_name = "rammp_odom3d"

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
    maintainer="Sarvesh Prajapati",
    maintainer_email="prajapati.s@northeastern.edu",
    description="World-anchored object tracking from ObjectDetections, aligned depth and odometry",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["odom3d_node = rammp_odom3d.node:main"]},
)
