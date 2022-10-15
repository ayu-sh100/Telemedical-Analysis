import setuptools

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''


setuptools.setup(
    name="PySimpleGUIWeb",
    version="0.39.0",
    author="PySimpleGUI",
    author_email="PySimpleGUI@PySimpleGUI.org",
    install_requires=['remi<=2020.3.10',],
    description="A port of PySimpleGUI that runs in a web browser.  Utilizes Remi as the GUI framework",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords="GUI UI Remi tkinter wrapper simple easy beginner novice student graphics progressbar progressmeter",
    url="https://github.com/PySimpleGUI/PySimpleGUI",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Topic :: Multimedia :: Graphics",
        "Operating System :: OS Independent"
    ),
)