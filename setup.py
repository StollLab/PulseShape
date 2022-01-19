from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='PulseShape',
    version='0.1.2-1',
    packages=['PulseShape'],
    python_requires='>=3.6',
    install_requires=['numpy>=1.19', 'scipy>=1.5'],
    url='https://gitlab.com/mtessmer/PulseShape',
    project_urls = {'Source': 'https://gitlab.com/mtessmer/PulseShape'},
    license='GNU GPLv3',
    license_files=('LICENSE'),
    author='Maxx Tessmer',
    author_email='mhtessmer@gmail.com',
    keywords=['pulse shape', 'EPR', 'DEER', 'PELDOR', 'AWG'],
    description='A pulse shaping program for EPR!',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    classifiers=['License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',]

)
