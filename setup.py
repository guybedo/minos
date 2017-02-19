from setuptools import setup
from setuptools import find_packages


setup(name='Minos',
      version='0.0.1',
      description='Search architecture & hyper parameters for neural nets',
      author='Julien Roch',
      author_email='julien.roch@akalea.com',
      url='https://github.com/guybedo/minos',
      license='MIT',
      install_requires=[
          'numpy>=1.12',
          'keras',
          'deap==1.1.0'],
      dependency_links=[
          "git+ssh://git@github.com/DEAP/deap.git@a1412d71b50606a7e4e87c3ba538b25603b84266#egg=deap-1.1.0"
      ],
      extras_require={
          'tests': ['pytest',
                    'pytest-cov',
                    'pytest-pep8',
                    'pytest-xdist',
                    'python-coveralls',
                    'coverage==3.7.1'],
      },
      packages=find_packages())
