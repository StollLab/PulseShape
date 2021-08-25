from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='PulseShape',
    version='0.1.3',
    packages=['PulseShape'],
    install_requires=['numpy>=1.19', 'scipy>=1.5', 'numba>=0.50'],
    url='https://gitlab.com/mtessmer/PulseShape',
    license='GNU GPLv3',
    license_files=('LICENSE')
    author='Maxx Tessmer',
    author_email='mhtessmer@gmail.com',
    description='A Pulse shaping program for EPR!',
    long_description = readme,
    long_description_content_type = 'text/markdown'

)
