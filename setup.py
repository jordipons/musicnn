from setuptools import setup, find_packages

with open('README.md') as file:
    long_description = file.read()

setup(
    name='musicnn',
    version='0.1.0',
    description='Pronounced as "musician", musicnn is a set of pre-trained deep convolutional neural networks for music audio tagging',
    author='Jordi Pons',
    url='http://github.com/jordipons/musicnn',
    packages=find_packages(),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='audio music deep learning tagging tensorflow machine listening',
    license='ISC',
    install_requires=['librosa>=0.7.0',
                      'tensorflow>=1.14',
                      'numpy<1.17,>=1.14.5']
)
