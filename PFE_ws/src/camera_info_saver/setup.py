from setuptools import find_packages, setup

package_name = 'camera_info_saver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zireael',
    maintainer_email='matisduval18magic@gmail.com',
    description='set_camera_info service for OAK-D calibration',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_info_saver = camera_info_saver.camera_info_saver:main',
        ],
    },
)
