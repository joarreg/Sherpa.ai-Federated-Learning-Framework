from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="shfl",
      version="0.1.0",
      description="Sherpa.FL is an open-source framework for Machine Learning that is dedicated to data privacy "
                  "protection",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/sherpaai/Sherpa.FL",
      packages=find_packages(),
      install_requires=['numpy', 'emnist', 'scikit-learn', 'pytest', 'tensorflow>=2.2.0', 'scipy', 'six', 'pathlib2'],
      python_requires='>=3.7')
