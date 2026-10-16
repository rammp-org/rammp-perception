from setuptools import setup

package_name = "rammp_detection_common"

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
    description="Shared class list, messages and node plumbing for the RAMMP perception detectors",
    license="Proprietary",
    tests_require=["pytest"],
)
