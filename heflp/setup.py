from setuptools import setup, find_packages
setup(
    name="heflp",
    version="0.1",
    description='An extension enabling several homomorphic encryption schemes on Flower FL framework',
    author='Zekun Wang',
    author_email='wangzekun.felix@gmail.com',
    packages=find_packages(),
    install_requires=[
        'flwr>=1.4',
        'numpy>=1.21.4',
        'pyfhel>=3.4.1',
        'tqdm'
    ]
)