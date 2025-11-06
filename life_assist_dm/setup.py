import os
from glob import glob
from setuptools import setup

package_name = 'life_assist_dm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='js',
    maintainer_email='moonjongsul@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'dialog_manager = life_assist_dm.dialog_manager.dialog_manager_node:main',
            'stt_node = life_assist_dm.sound.stt_node:main',
            'tts_node = life_assist_dm.sound.tts_node:main',
        ],
    },
)
