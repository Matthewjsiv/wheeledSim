import setuptools
from setuptools import find_packages

setuptools.setup(
	name = 'wheeledSim',
	packages=find_packages() + ['wheeledRobots.clifford', 'wheeledRobots.simple_car'],
	include_package_data=True,
	package_data={"":['*.sdf','meshes/*', '*.urdf']}
)
