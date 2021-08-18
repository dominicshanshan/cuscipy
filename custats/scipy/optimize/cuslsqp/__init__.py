import os
import subprocess
import pathlib


def compile_helper(glob_regx: str):
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process.
    glob_regx (str): string passed to glob.glob for selecting files to build
    """
    current = os.getcwd()
    path = os.path.abspath(os.path.dirname(__file__))
    build_bfgs = len(list(pathlib.Path(path).glob(glob_regx))) > 0
    if build_bfgs:
        return
    os.chdir(path)
    ret = subprocess.run(['python3', 'setup.py', 'build_ext', '-i'])
    if ret.returncode != 0:
        print("build sys module failed, exiting.")
        import sys
        sys.exit(1)
    os.chdir(current)


BUILD_FILES = 'slsqp*.so'
compile_helper(BUILD_FILES)
