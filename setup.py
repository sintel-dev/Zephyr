#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

try:
    with open('README.md') as readme_file:
        readme = readme_file.read()
except IOError:
    readme = ''

try:
    with open('HISTORY.md') as history_file:
        history = history_file.read()
except IOError:
    history = ''

install_requires = [
    'numpy>=1.19.5,<1.27.0',
    'pandas>=1,<2',
    'composeml>=0.1.6,<0.10',
    'featuretools>=1.0.0,<2.0.0',
    'mlblocks>=0.6.0,<0.7',
    'sigpro>=0.2.0',
    'xgboost>=0.72.1,<1',
    'jupyter==1.0.0',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'jupyter>=1.0.0,<2',
    'rundoc>=0.4.3,<0.5',
    'invoke',
]

development_requires = [
    # general
    'bumpversion>=0.5.3,<0.6',
    'pip>=9.0.1',
    'watchdog>=0.8.3,<0.11',

    # docs
    'docutils>=0.12,<0.18',
    'lxml_html_clean>=0.2.2',
    'm2r2>=0.2.5,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx>=3,<3.3',
    'sphinx_rtd_theme>=0.2.4,<0.5',
    'autodocsumm>=0.1.10',
    'mistune>=0.7,<2.0',
    'Jinja2>=2,<3.1',

    # fails on Sphinx < v3.4
    'alabaster<=0.7.12',
    # fails on Sphins < v5.0
    'sphinxcontrib-applehelp<1.0.8',
    'sphinxcontrib-devhelp<1.0.6',
    'sphinxcontrib-htmlhelp<2.0.5',
    'sphinxcontrib-serializinghtml<1.1.10',
    'sphinxcontrib-qthelp<1.0.7',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',

    # Jupyter
    'jupyter>=1.0.0',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dai-lab@mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description='Prediction engineering methods for Draco.',
    entry_points={
        'mlblocks': [
            'primitives=zephyr_ml:MLBLOCKS_PRIMITIVES',
            'pipelines=zephyr_ml:MLBLOCKS_PIPELINES'
        ],
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='zephyr Draco Prediction Engineering',
    name='zephyr-ml',
    packages=find_packages(include=['zephyr_ml', 'zephyr_ml.*']),
    python_requires='>=3.8,<3.13',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sintel-dev/zephyr',
    version='0.0.5.dev0',
    zip_safe=False,
)
