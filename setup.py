import os
import sys
import sysconfig
import importlib
import subprocess
import glob
from distutils.command.build import build
from distutils.command.clean import clean
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.develop import develop
from setuptools import setup

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pyrecon'

sys.path.insert(0,os.path.abspath(package_basename))
import _version
import utils
version = _version.__version__
lib_dir = utils.lib_dir
src_dir = os.path.join(package_basedir,'src')


def find_compiler():
    compiler = os.getenv('CC',None)
    if compiler is None:
        compiler = sysconfig.get_config_vars().get('CC',None)
    return compiler


class custom_build(build):

    def run(self):
        super(custom_build,self).run()

        os.environ.setdefault('LIBDIR',lib_dir)
        library_dir = sysconfig.get_config_var('LIBDIR')
        os.environ.setdefault('OMPFLAG','-fopenmp -L{}'.format(library_dir))

        compiler = find_compiler()
        if compiler == 'clang':
            os.environ.setdefault('CC','clang')
            os.environ.setdefault('OMPFLAG','-Xclang -fopenmp -L{}'.format(library_dir))

        def compile():
            subprocess.call('make',shell=True,cwd=src_dir)

        self.execute(compile,[],'Compiling')


class custom_bdist_egg(bdist_egg):

    def run(self):
        self.run_command('build')
        super(custom_bdist_egg,self).run()


class custom_develop(develop):

    def run(self):
        self.run_command('build')
        super(custom_develop,self).run()


class custom_clean(clean):

    def run(self):
        # run the built-in clean
        super(custom_clean,self).run()
        # remove the recon products
        os.environ.setdefault('LIBDIR',lib_dir)
        subprocess.call('make clean',shell=True,cwd=src_dir)


if __name__ == '__main__':

    setup(name=package_basename,
          version=version,
          author="Arnaud de Mattia",
          author_email='',
          description="Python wrapper for reconstruction codes",
          license='GPL3',
          url='http://github.com/adematti/pyrecon',
          install_requires=['numpy'],
          cmdclass = {
              'build': custom_build,
              'develop': custom_develop,
              'bdist_egg': custom_bdist_egg,
              'clean': custom_clean
          },
         data_files = [('lib',glob.glob('lib/*.so'))],
         packages=[package_basename]
    )
