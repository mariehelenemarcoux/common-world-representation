from setuptools import setup, find_packages

setup(
    name="cwr",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["torch>=2.0.0", "numpy>=1.24.0"],
)
