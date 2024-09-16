# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os import path
import re

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

root_dir = path.abspath(path.dirname(__file__))
package_name = "pydisconet"
version_file = path.join(root_dir, 'VERSION')

with open(path.join(root_dir, 'src', package_name, '__init__.py')) as f:
    init_text = f.read()
    license = re.search(r'__license__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author = re.search(r'__author__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author_email = re.search(r'__author_email__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    url = re.search(r'__url__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

with open(path.join(root_dir, 'src', package_name, 'requirements.txt')) as f:
    required = f.read().splitlines()
# Start install process
setup(
    setup_requires=["setuptools-git-versioning>=2.0,<3"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": version_file,
    },
    name=package_name,
    # version=version, #
    description='analyzing the co-authorship network of researchers in the field of biology',
    long_description=readme,
    keywords='co-authorship, ML, AI, network analysis, graph theory, biology, bioinformatics',
    python_requires='>=3.8',
    classifiers=[# How mature is this project? Common values are
                #   3 - Alpha
                #   4 - Beta
                #   5 - Production/Stable
                'Development Status :: 3 - Alpha',

                # Indicate who your project is intended for
                'Intended Audience :: Developers',
                'Topic :: Software Development :: Build Tools',

                # Specify the Python versions you support here. In particular, ensure
                # that you indicate whether you support Python 2, Python 3 or both.
                'Programming Language :: Python :: 3.10',
            ],
    install_requires=required,
    author=author,
    author_email=author_email,
    url=url,
    license=license,
    package_data={"pydisconet": ["scripts/*", "notebooks/*", "input.yaml", "0.parse.sh", "requirements.txt"]},
    packages=["pydisconet", "pydisconet.analyze", "pydisconet.database_parser", "pydisconet.plotter", "pydisconet.preprocessing", "pydisconet.utils"],
)