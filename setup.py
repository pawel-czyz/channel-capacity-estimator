from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='channel-capacity-estimator',
    version='1.0.1',
    description='Package for estimation of information channel capacity.',
    url='http://pmbm.ippt.pan.pl/software/cce',
    author='Frederic Grabowski, Pawel Czyz',
    author_email='grabowski.frederic@gmail.com, pczyz@protonmail.com',
    license='GNU GPL 3.0 license',
    packages=['cce'],
    install_requires=['numpy', 'scipy', 'tensorflow'],

    # if anybody wants to add non-py files, s/he must uncomment the following line and add them in MANIFEST.in
    # include_package_data=True
    zip_safe=False
)
