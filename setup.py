from setuptools import setup, find_packages

setup(
    name="cinematch",
    version="1.0.0",
    description="A hybrid movie recommender system using Collaborative Filtering, Matrix Factorization, and Content-Based Filtering.",
    author="Your Name",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter", "matplotlib", "seaborn"],
    },
)
