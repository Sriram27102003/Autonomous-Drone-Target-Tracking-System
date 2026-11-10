# Used by catkin_python_setup() only; do not run directly.
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["human_detector"],
    package_dir={"": "src"},
)

setup(**setup_args)
