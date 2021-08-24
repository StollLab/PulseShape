from setuptools import setup

setup(
    name='PulseShape',
    version='0.1.0',
    packages=['PulseShape'],
    install_requires=['numpy>=1.19', 'scipy>=1.5', 'numba>=0.50'],
    url='https://gitlab.com/mtessmer/PulseShape',
    license='GNU GPLv2',
    author='Maxx Tessmer',
    author_email='mhtessmer@gmail.com',
    description='A Pulse shaping program for EPR!'
)
