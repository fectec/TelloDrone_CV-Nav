from setuptools import find_packages, setup

package_name = 'tello_driver'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fectec',
    maintainer_email='fectec151@gmail.com',
    description='Driver for the RoboMaster TT Tello Talent Drone.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tello_driver_node = tello_driver.tello_driver_node:main',
        ],
    },
)