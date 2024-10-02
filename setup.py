from setuptools import setup

setup(
    name="stlcgpp",
    version="0.0.1",
    description="stlcg++ with pytorch",
    author="Karen Leung",
    author_email="kymleung@uw.edu",
    packages=["stlcgpp"],
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "graphviz"
    ],
)