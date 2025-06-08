from setuptools import setup

package_name = 'tello_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fectec',
    maintainer_email='fectec151@gmail.com',
    description='Computer vision algorithms for the RoboMaster TT Tello Talent Drone.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tello_downward_camera = tello_vision.tello_downward_camera:main',
            'aruco_detector = tello_vision.aruco_detector:main',
        ],
    },
)