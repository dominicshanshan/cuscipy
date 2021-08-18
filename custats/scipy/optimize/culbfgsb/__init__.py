import os
import subprocess
import multiprocessing
import pathlib


def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""
    cpus = multiprocessing.cpu_count()
    current = os.getcwd()
    path = os.path.abspath(os.path.dirname(__file__))
    build_bfgs = len(list(pathlib.Path(path).glob('bfgs*.so'))) > 0
    build_cudalib = len(list(
        pathlib.Path(path).glob('build/culbfgsb/*.so'))) > 0
    if build_bfgs and build_cudalib:
        return
    # os.environ['LD_LIBRARY_PATH'] = path+'/build/culbfgsb/'
    ret = subprocess.run(['mkdir', '-p', path+'/build'])
    if ret.returncode != 0:
        print("Create dir {} failed, exiting.".format(path+'/build'))
        import sys
        sys.exit(1)
    os.chdir(path+'/build')
    print(os.getcwd())
    ret = subprocess.run(['cmake', '..'])
    if ret.returncode != 0:
        print("cmake failed, exiting.")
        import sys
        sys.exit(1)
    ret = subprocess.run(['make', '-j', str(cpus)])
    if ret.returncode != 0:
        print("cmake failed, exiting.")
        import sys
        sys.exit(1)
    os.chdir(path)
    ret = subprocess.run(['python3', 'setup.py', 'build_ext', '-i'])
    if ret.returncode != 0:
        print("build lbfgsb module failed, exiting.")
        import sys
        sys.exit(1)
    os.chdir(current)


compile_helper()
from .bfgs import fmin
