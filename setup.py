from setuptools import setup, find_packages

setup(
  name = 'VN-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.6',
  license='MIT',
  description = 'Vector Neuron Transformer (VN-Transformer)',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/VN-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'equivariance',
    'vector neurons',
    'transformers',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.6.0',
    'torch>=1.6'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)
