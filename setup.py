from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='channel-capacity-estimator',
    version='1.0.0',
    description='Package estimating channel capacity.',
    url='todo',
    author='Frederic Grabowski, Pawel Czyz, Marek Kochanczyk, Tomasz Lipniacki',
    author_email='grabowski.frederic@gmail.com, pczyz@protonmail.com',
    license='BSD 3-Clause License',
    packages=['cce'],
    install_requires=['numpy', 'tensorflow', 'scipy'],

    # if anybody wants to add non-py files, he must uncomment the following line and add them in MANIFEST.in
    # include_package_data=True
    zip_safe=False
)
