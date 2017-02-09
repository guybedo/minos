from setuptools import setup
from setuptools import find_packages


setup(name='Minos',
      version='0.0.1',
      description='Search architecture & hyper parameters for neural nets',
      author='Julien Roch',
      author_email='julien.roch@akalea.com',
      url='https://github.com/guybedo/minos',
      download_url='https://github.com/fchollet/keras/tarball/1.2.1',
      license='MIT',
      install_requires=[
          'numpy',
          'keras',
          'deap>=1.0.2'],
      extras_require={
          'tests': ['pytest',
                    'pytest-cov',
                    'pytest-pep8',
                    'pytest-xdist',
                    'python-coveralls',
                    'coverage==3.7.1'],
      },
      packages=find_packages())
