from setuptools import setup, find_packages
import os

# Get the directory containing this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="surgical_robot_project",
    version="0.1.0",
    description="Surgical Robot Reinforcement Learning with Isaac Lab",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
    },
)