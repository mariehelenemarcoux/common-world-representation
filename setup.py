from setuptools import setup, find_packages

setup(
    name="cwr",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
