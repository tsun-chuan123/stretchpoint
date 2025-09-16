import sys
from setuptools import setup, find_packages
import os
from glob import glob

# 移除 colcon 傳遞的不支援選項
unsupported_options = ['--build-directory', '--install-directory', '--editable']
for option in unsupported_options:
    if option in sys.argv:
        try:
            idx = sys.argv.index(option)
            sys.argv.pop(idx)  # 移除選項
            # 檢查下一個參數是否為值（不以 -- 開頭）
            if idx < len(sys.argv) and not sys.argv[idx].startswith('--'):
                sys.argv.pop(idx)  # 移除值
        except ValueError:
            pass

package_name = 'd435i_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stretch_developer',
    maintainer_email='your_email@example.com',
    description='D435i camera node for Stretch3 VLA system',
    license='MIT',
    test_suite='pytest',
    entry_points={
        'console_scripts': [
            'camera_node = d435i_camera.camera_node:main',
        ],
    },
)
