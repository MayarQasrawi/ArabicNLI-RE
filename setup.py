from setuptools import find_packages, setup

setup(
    name="arabic-nli-classifier",
    version="1.0.0",
    description="Arabic NLI sentence classifier using ARBERTv2 with contrastive loss.",
    author="Alaa Aljabari",
    author_email="aaljabari@birzeit.edu",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "openpyxl>=3.1.2",
        "numpy>=1.26.0",
        "more-itertools>=10.2.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
