import os
import sys
import sysconfig
import subprocess
import shutil
from distutils.command.build import build
from distutils.command.clean import clean
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.develop import develop
from setuptools import setup

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pyrecon'

sys.path.insert(0, os.path.join(package_basedir, package_basename))
import _version
import utils
version = _version.__version__
lib_dir = utils.lib_dir
src_dir = os.path.join(package_basedir, 'src')


def find_compiler():
    compiler = os.getenv('CC', None)
    if compiler is None:
        compiler = sysconfig.get_config_vars().get('CC', None)
    import platform
    uname = platform.uname().system
    if compiler is None:
        compiler = 'gcc'
        if uname == 'Darwin': compiler = 'clang'
    return compiler


def compiler_is_clang(compiler):
    if compiler == 'clang':
        return True
    from subprocess import Popen, PIPE
    proc = Popen([compiler, '--version'], universal_newlines=True, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    if 'clang' in out:
        return True
    return False


class custom_build(build):

    def run(self):
        super(custom_build, self).run()

        # lib_dir = os.path.join(os.path.abspath(self.build_lib),'pyrecon','lib')
        os.environ.setdefault('LIBDIR', lib_dir)
        library_dir = sysconfig.get_config_var('LIBDIR')

        compiler = find_compiler()
        os.environ.setdefault('CC', compiler)
        if compiler_is_clang(compiler):
            flags = '-Xclang -fopenmp -L{} -lomp'.format(library_dir)
        elif compiler in ['cc', 'icc']:
            flags = '-fopenmp -L{} -lgomp -limf -liomp5'.format(library_dir)
        else:
            flags = '-fopenmp -L{} -lgomp'.format(library_dir)
        os.environ.setdefault('OMPFLAG', flags)

        def compile():
            subprocess.call('make', shell=True, cwd=src_dir)

        self.execute(compile, [], 'Compiling')
        new_lib_dir = os.path.join(os.path.abspath(self.build_lib), package_basename, 'lib')
        shutil.rmtree(new_lib_dir, ignore_errors=True)
        shutil.copytree(lib_dir, new_lib_dir)


class custom_bdist_egg(bdist_egg):

    def run(self):
        self.run_command('build')
        super(custom_bdist_egg, self).run()


class custom_develop(develop):

    def run(self):
        self.run_command('build')
        super(custom_develop, self).run()


class custom_clean(clean):

    def run(self):
        # run the built-in clean
        super(custom_clean, self).run()
        # remove the recon products
        shutil.rmtree(lib_dir, ignore_errors=True)


if __name__ == '__main__':

    setup(name=package_basename,
          version=version,
          author='cosmodesi',
          author_email='',
          description='Python wrapper for reconstruction codes',
          license='BSD3',
          url='http://github.com/cosmodesi/pyrecon',
          install_requires=['numpy'],
          extras_require={'extras': ['fitsio', 'h5py', 'astropy'], 'metrics': ['pypower @ git+https://github.com/cosmodesi/pypower']},
          cmdclass={
              'build': custom_build,
              'develop': custom_develop,
              'bdist_egg': custom_bdist_egg,
              'clean': custom_clean
          },
          packages=[package_basename],
          scripts=['bin/pyrecon'])
