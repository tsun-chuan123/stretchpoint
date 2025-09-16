from setuptools import setup, find_packages

package_name = 'llava_server'

setup(
    name='llava-server',
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/llava_server.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stretch_developer',
    maintainer_email='your_email@example.com',
    description='LLaVA server for Stretch3 VLA system',
    license='MIT',
    entry_points={
        'console_scripts': [
            'server_node = llava_server.server_node:main',
        ],
    },
)
