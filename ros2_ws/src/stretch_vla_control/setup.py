import sys
from setuptools import setup

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

package_name = 'stretch_vla_control'

setup(
    name=package_name,
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/control_launch.py']),
        ('share/' + package_name + '/config', ['config/robot_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stretch_developer',
    maintainer_email='your_email@example.com',
    description='Stretch VLA control system',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'control_node = stretch_vla_control.control_node:main',
        ],
    },
)
