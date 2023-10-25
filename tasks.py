import glob
import operator
import os
import platform
import re
import shutil
import stat
from pathlib import Path

from invoke import task


COMPARISONS = {
    '>=': operator.ge,
    '>': operator.gt,
    '<': operator.lt,
    '<=': operator.le
}


@task
def pytest(c):
    c.run('python -m pytest --cov=zephyr_ml')


@task
def install_minimum(c):
    with open('setup.py', 'r') as setup_py:
        lines = setup_py.read().splitlines()

    versions = []
    started = False
    for line in lines:
        if started:
            if line == ']':
                started = False
                continue

            line = line.strip()
            if line.startswith('#'): # ignore comment
                continue

            # get specific package version based on declared python version
            if 'python_version' in line:
                python_version = re.search(r"python_version(<=?|>=?)\'(\d\.?)+\'", line)
                operation = python_version.group(1)
                version_number = python_version.group(0).split(operation)[-1].replace("'", "")
                if COMPARISONS[operation](platform.python_version(), version_number):
                    line = line.split(";")[0]
                else:
                    continue
                
            line = re.sub(r"""['",]""", '', line)
            line = re.sub(r'>=?', '==', line)
            if '==' in line:
                line = re.sub(r',?<=?[\d.]*,?', '', line)
            
            elif re.search(r',?<=?[\d.]*,?', line):
                line = f"'{line}'"

            versions.append(line)

        elif line.startswith('install_requires = ['):
            started = True

    c.run(f'python -m pip install {" ".join(versions)}')


@task
def minimum(c):
    install_minimum(c)
    c.run('python -m pip check')
    c.run('python -m pytest')


@task
def readme(c):
    test_path = Path('tests/readme_test')
    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)

    cwd = os.getcwd()
    os.makedirs(test_path, exist_ok=True)
    shutil.copy('README.md', test_path / 'README.md')
    shutil.copytree('notebooks/data', test_path / 'notebooks/data')
    os.chdir(test_path)
    c.run('rundoc run --single-session python3 -t python3 README.md')
    os.chdir(cwd)
    shutil.rmtree(test_path)


@task
def tutorials(c):
    for ipynb_file in glob.glob('notebooks/*.ipynb'):
        if '.ipynb_checkpoints' not in ipynb_file:
            c.run((
                'jupyter nbconvert --execute --ExecutePreprocessor.timeout=3600 '
                f'--to=html --stdout {ipynb_file} > /dev/null'
            ), hide='out')


@task
def lint(c):
    c.run('flake8 zephyr_ml tests')
    c.run('isort -c --recursive zephyr_ml tests')


@task
def checkdeps(c, path):
    with open('setup.py', 'r') as setup_py:
        lines = setup_py.read().splitlines()

    packages = []
    started = False
    for line in lines:
        if started:
            if line == ']':
                started = False
                continue

            line = line.strip()
            if line.startswith('#') or not line: # ignore comment
                continue
                
            line = re.split(r'>?=?<?', line)[0]
            line = re.sub(r"""['",]""", '', line)
            packages.append(line)

        elif line.startswith('install_requires = ['):
            started = True

    c.run(
        f'pip freeze | grep -v \"sintel-dev/Zephyr.git\" | '
        f'grep -E \'{"|".join(packages)}\' > {path}'
    )


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)


@task
def rmdir(c, path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
    except PermissionError:
        pass