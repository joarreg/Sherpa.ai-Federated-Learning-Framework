from setuptools import setup, find_packages

setup(name="shfl", packages=find_packages(), install_requires=['numpy', 'emnist', 'scikit-learn', 'pytest',
                                                               'tensorflow>=2.2.0', 'scipy'])
