from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'RL_local_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
        os.path.join('share', 'RL_local_planner', 'launch'),
        glob('launch/*.launch.py')
    ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hkumar456',
    maintainer_email='hkumar456@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simulator_node = RL_local_planner.env_simulator:main',
            'controller_node = RL_local_planner.controller:main',
        ],
    },
)
