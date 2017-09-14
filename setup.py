from setuptools import find_packages
from setuptools import setup


setup(name='pysisy',
      version='0.1',
      description='Graphical UI and training shorthand for neural nets hyper parameters & architecture search with genetic algorithms',
      keywords=['keras', 'genetic algorithm', 'neural network', 'deep learning'],
      author='Charlie Sanders originally Julien Roch',
      author_email='charlie.fats@gmail.com',
      url='https://github.com/qorrect/minos',
      license='Apache',
      setup_requires=[
          'numpy>=1.12'],
      install_requires=[
          'numpy>=1.12',
          'keras>=2.0.0',
          'deap>=1.0.2'],
      extras_require={
          'h5py': ['h5py'],
          'tests': ['pytest',
                    'pytest-cov',
                    'pytest-pep8',
                    'pytest-xdist',
                    'python-coveralls',
                    'coverage==3.7.1'],
          'tf': ["tensorflow>=1.0.0"],
          'tf_gpu': ["tensorflow-gpu>=1.0.0"]
      },
      packages=find_packages())
