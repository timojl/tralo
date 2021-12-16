from setuptools import setup, find_packages

setup(
    name='tralo',
    entry_points={
    'console_scripts': [
        'tralo=tralo.cli:main',
    ]},
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.1',
    ],
)