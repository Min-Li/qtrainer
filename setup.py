from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="qtrainer",
    py_modules=["qtrainer"],
    version="0.0.1",
    description="A high-level API for training variational quantum circuits with error mitigation",
    author="Min Li, Haoxiang Wang",
    author_email="minl2@illinois.edu, hwang264@illinois.edu",
    url="https://github.com/Min-Li/qtrainer",
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Windows",
    ],
)
