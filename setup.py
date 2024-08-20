import pathlib
import setuptools


def read(fname: pathlib.Path):
    directory_name = pathlib.Path(__file__).parent
    return open(directory_name / fname, "r", encoding="utf8").read()


def read_requirements(fname: pathlib.Path):
    return read(fname).strip().split("\n")


setuptools.setup(
    name="maxent_deep_irl",
    version="0.0.1",
    description="PyTorch implementation of Maximum Entropy Deep Inverse Reinforcement Learning",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords="reinforcement learning, inverse reinforcement learning, imitation learning",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=["tests", "tests.*"]),
    install_requires=read_requirements("requirements.txt")
)
