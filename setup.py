from setuptools import setup, find_packages

setup(
    name='sreg',
    version='0.1.0',
    description='Stratified Randomized Experiments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Juri Trifonov, Yuehao Bai, Azeem Shaikh, Max Tabord-Meehan',
    author_email='jutrifonov@uchicago.edu',
    url='https://github.com/jutrifonov/sreg.py',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your package dependencies here
         'numpy>=1.15.0',
         'pandas>=0.23.0',
         'scipy>=1.1.0',
         'statsmodels>=0.14.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)


