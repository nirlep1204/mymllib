from setuptools import setup, find_packages

setup(
    name="mymllib",
    version="0.1.0",
    description="A personal machine learning library built from scratch.",
    author="Nirlep Makwana",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
    ],
    entry_points={
        "console_scripts": [
            "mymllib=mymllib.__main__:main",
        ],
    },
)
