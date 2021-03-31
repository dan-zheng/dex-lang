from setuptools import setup, find_packages
import os
import sys

def shared_library(name):
  """Returns a full shared library name with a platform-specific extension."""
  if sys.platform == 'darwin':
    return f"{name}.dylib"
  return f"{name}.so"

# Check that Dex shared library exists in the python/dex directory.
so_file = "libDex.so"
dex_dir = os.path.join(os.path.dirname(__file__), 'dex')
if not os.path.exists(os.path.join(dex_dir, so_file)):
  raise FileNotFoundError(f"{so_file} not found in python/dex/; "
                           "please run `make build-python`")
setup(
  name='dex',
  version='0.0.1',
  description='A research language for typed, functional array processing',
  license='BSD',
  author='Adam Paszke',
  author_email='apaszke@google.com',
  packages=find_packages(),
  package_data={'dex': [shared_library('libDex')]},
  install_requires=['numpy'],
)
  
