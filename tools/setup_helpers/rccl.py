import os
import glob
import warnings
from itertools import chain

from .env import IS_WINDOWS, IS_DARWIN, IS_CONDA, CONDA_DIR, check_negative_env_flag, \
    gather_paths

from .rocm import USE_ROCM, ROCM_HOME

USE_RCCL = False
RCCL_LIB_DIR = None
RCCL_SYSTEM_LIB = None
RCCL_INCLUDE_DIR = None
RCCL_ROOT_DIR = None
LIBRCCL_PREFIX = "librccl"
if USE_ROCM and not IS_DARWIN and not IS_WINDOWS and not check_negative_env_flag('USE_RCCL'):
   ENV_ROOT = os.getenv('RCCL_ROOT_DIR', None)
   LIB_DIR = os.getenv('RCCL_LIB_DIR', None)
   INCLUDE_DIR = os.getenv('RCCL_INCLUDE_DIR', None)

   lib_paths = list(filter(bool, [
       LIB_DIR,
       ENV_ROOT,
       os.path.join(ENV_ROOT, 'lib') if ENV_ROOT is not None else None,
       os.path.join(ENV_ROOT, 'lib', 'x86_64-linux-gnu') if ENV_ROOT is not None else None,
       os.path.join(ENV_ROOT, 'lib64') if ENV_ROOT is not None else None,
       os.path.join(ROCM_HOME, 'lib'),
       os.path.join(ROCM_HOME, 'lib64'),
       '/usr/local/lib',
       '/usr/lib/x86_64-linux-gnu/',
       '/usr/lib/powerpc64le-linux-gnu/',
       '/usr/lib/aarch64-linux-gnu/',
       '/usr/lib',
   ] + gather_paths([
       'LIBRARY_PATH',
   ]) + gather_paths([
       'LD_LIBRARY_PATH',
   ])))

   include_paths = list(filter(bool, [
       INCLUDE_DIR,
       ENV_ROOT,
       os.path.join(ENV_ROOT, 'include') if ENV_ROOT is not None else None,
       os.path.join(ROCM_HOME, 'include'),
       '/usr/local/include',
       '/usr/include',
   ]))

   if IS_CONDA:
       lib_paths.append(os.path.join(CONDA_DIR, 'lib'))
   for path in lib_paths:
       path = os.path.expanduser(path)
       if path is None or not os.path.exists(path):
           continue
       if glob.glob(os.path.join(path, LIBRCCL_PREFIX + '*')):
           RCCL_LIB_DIR = path
           # try to find an exact versioned .so/.dylib, rather than librccl.so
           preferred_path = glob.glob(os.path.join(path, LIBRCCL_PREFIX + '*[0-9]*'))
           if len(preferred_path) == 0:
               RCCL_SYSTEM_LIB = glob.glob(os.path.join(path, LIBRCCL_PREFIX + '*'))[0]
           else:
               RCCL_SYSTEM_LIB = os.path.realpath(preferred_path[0])
           break
   for path in include_paths:
       path = os.path.expanduser(path)
       if path is None or not os.path.exists(path):
           continue
       if glob.glob(os.path.join(path, 'rccl.h')):
           RCCL_INCLUDE_DIR = path
           break

   if RCCL_LIB_DIR is not None and RCCL_INCLUDE_DIR is not None:
       USE_RCCL = True
       RCCL_ROOT_DIR = os.path.commonprefix((RCCL_LIB_DIR, RCCL_INCLUDE_DIR))
