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

package_name = 'stretch_gui'

setup(
    name=package_name,
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stretch_developer',
    maintainer_email='your_email@example.com',
    description='PyQt5 GUI for Stretch3 VLA system',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'gui_node = stretch_gui.gui_node:main',
            'send_camera_images = stretch_gui.send_camera_images:main',
            'vla_command_test = stretch_gui.vla_command_test:main',
        ],
    },
)
