from setuptools import setup

setup(name='up',
      version='0.1',
      description='Universal pretraining',
      url='peterbloem.nl',
      author='Peter Bloem',
      author_email='up@peterbloem.nl',
      license='MIT',
      packages=['up'],
      install_requires=[
            'torch',
            'tqdm',
            'numpy',
            'torchtext',
            'fire',
            'pyyaml',
            'wandb',
            'former' # https://github.com/pbloem/former
      ],
      zip_safe=False)