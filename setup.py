import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='ObstacleModulation',
    version='1.0.0',
    description='Dynamic Obstacle Avoidance through Obstacle Modulation to stretch space around objects to not collide with them and arrive at the goal.',
    long_description=readme,
    author='Alexandre Coulombe',
    author_email='alexandre.coulombe@mail.mcgill.ca',
    packages=setuptools.find_packages(exclude=('examples', )),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'Collision @ git+https://github.com/acoulombe/CollisionLibrary.git@main#egg=Collision'],
)
