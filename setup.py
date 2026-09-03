from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

setup(
    name='sreg',
    version='2.1.0',
    description='Stratified Randomized Experiments',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Juri Trifonov, Yuehao Bai, Azeem Shaikh, Max Tabord-Meehan',
    author_email='jutrifonov@u.northwestern.edu',
    url='https://github.com/jutrifonov/sreg_py',
    project_urls={
        'Documentation': 'https://github.com/jutrifonov/sreg_py/tree/main/docs',
        'Source': 'https://github.com/jutrifonov/sreg_py',
        'Issues': 'https://github.com/jutrifonov/sreg_py/issues',
        'Changelog': 'https://github.com/jutrifonov/sreg_py/blob/main/CHANGELOG.md',
    },
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'sreg': ['data/AEJapp.csv'],
    },
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.23',
        'pandas>=1.5',
        'scipy>=1.9',
        'statsmodels>=0.14.2',
        'matplotlib>=3.7',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    license='MIT',
    license_files=('LICENSE',),
    keywords='causal-inference experiments randomization stratification',
    python_requires='>=3.9',
)
