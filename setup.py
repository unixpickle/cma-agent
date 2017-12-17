"""
Package meta-data.
"""

from setuptools import setup

setup(
    name='cma-agent',
    version='0.1.0',
    description='A reinforcement learning agent that uses CMA-ES.',
    long_description='A reinforcement learning agent that uses CMA-ES.',
    url='https://github.com/unixpickle/cma-agent',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='MIT',
    packages=['cma_agent'],
    install_requires=['anyrl>=0.10.4<0.12.0', 'cma>=2.3.0<3.0.0']
)
