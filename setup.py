from setuptools import setup

package_name = 'ai_utils'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[],
    install_requires=['setuptools', 'opencv-python', 'mediapipe', 'matplotlib', 'torch', 'sklearn'],
    zip_safe=True,
    maintainer='Federico Rollo',
    maintainer_email='rollo.f96@gmail.com',
    description='This package contains functions useful for ML algorithm use. It creates simple interface between some used AI algorithms.',
    license='GNU GENERAL PUBLIC LICENSE v3',
    entry_points={},
)
