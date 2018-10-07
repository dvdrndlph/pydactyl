import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='pydactyl',
    version='0.5',
    description='Toolkit for piano fingering. Part of Didactyl project at the University of Illinois at Chicago.',
    url='http://github.com/dvdrndlph/pydactyl',
    author='David A. Randolph',
    author_email='drando2@uic.edu',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'music21>=5.3',
        'pymysql>=0.8',
        'networkx>=2.1',
        'TatSu',
    ],
)
