# Table of Content

0. [What and why?](#0-whatNwhy)
1. [Python Installation](#1-installation)
    1. [Linux](#1-1-Linux)
    2. [Windows](#1-2-windows)
    3. [MacOS](#1-3-macos)
2. [Virtual Environment](#2-virtualEnvironment)
    1. [Linux](#2-1-Linux)
    2. [Windows](#2-2-windows)
    3. [MacOS](#2-3-macos)
    4. [Virtual Environment in action](#2-4-action)
3. [Jupyter](#3-jupyter)
# <a name="0-whatNwhy"></a>0. What and Why
### What is Python? 
[quote](https://www.w3schools.com/python/python_intro.asp)

Python is a popular programming language. It was created by Guido van Rossum, and released in 1991.

It is used for:

- web development (server-side),
- software development,
- mathematics,
- system scripting.

### Why uses Python?
[quote](https://www.w3schools.com/python/python_intro.asp)
- Python works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc).
- Python has a simple syntax similar to the English language.
- Python has syntax that allows developers to write programs with fewer lines than some other programming languages.
- Python runs on an interpreter system, meaning that code can be executed as soon as it is written. This means that prototyping can be very quick.
- Python can be treated in a procedural way, an object-orientated way or a - functional way.

### Before you start
1. There are two major versions of Python which are Python2 and Python3. Note that when you call `python` in your terminal, you are referring to Python2. Instead, you need to explicit version 3 to call to Python3 (`python3`).
2. There are different syntax between the two versions. We are recommended that you use version 3 throughout the course.

# <a name="1-installation"></a>1. Python Installation
## <a name="1-1-Linux"></a>1. For Linux user

[Reference](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-an-Linux-20-04-server)

### Install via `apt`
1. Open your terminal
2. Update and Upgrade your current packages.
```
$ sudo apt update
$ sudo apt upgrade
```
3. Install `python3`
```
$ sudo apt install python3
```
4. Verify that you have `python3`
```
$ python3 -V
Python 3.8.2
```
5. You would need to install a package/library/module downloader. Similar to Python, there are `pip` for Python2 and `pip3` for Python3.
```
$ sudo apt install python3-pip
```
6. Verify thar you have `pip3`
```
$ pip3 -V
pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)
```
7. Let's try to install your first package `numpy`
```
$ pip3 install numpy
Collecting numpy
  Using cached numpy-1.18.4-cp38-cp38-manylinux1_x86_64.whl (20.7 MB)
Installing collected packages: numpy
Successfully installed numpy-1.18.4
```

*Note that we are not using `sudo` in front of the command. When you are installing the Linux package, it is required to use root privilege here. Using `sudo` in front of any command to run the command as root. However, when you are installing the python package, you do not need to issue root privilege. In fact, doing so is not recommended due to the security concerned.*

8. Install development tools.
```
$ sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

## <a name="1-2-windows"></a>2. For Windows user

### Installer from python.org
1. Download 64-bit installer [link](https://www.python.org/ftp/python/3.8.3/python-3.8.3-amd64.exe). You need 64-bit version so that you can run PyTorch.

2. Check `Add Python 3.8 to PATH` option and `Install Now`.

![alt](https://raw.githubusercontent.com/chaklam-silpasuwanchai/Python-for-DS-AI/master/.0%20-%20installation_image/python-windows-install-64.png)

3. Veryify that `python` is installed with `pip` using your `cmd`
```
C:>python -V
Python 3.8.3

C:>pip -V
pip 20.1.1 from c:\users\<username>\appdata\local\programs\python\python38-32\lib\site-packages\pip (python 3.8)
```
4. Let's try to install your first package `numpy`
```
C:>pip install numpy
Collecting numpy
  Downloading https://files.pythonhosted.org/packages/8a/52/daf6f4b7fd1499c153cb25ff84f87421598d95e5bb5b760585d2c0263773/numpy-1.18.5-cp38-cp38-win32.whl (10.8MB)
     |████████████████████████████████| 10.8MB 6.4MB/s
Installing collected packages: numpy
Successfully installed numpy-1.18.5
```

## <a name="1-3-macos"></a>3. For MacOS user

Some caution: Do not use the pre-installed python from MacOS.  That's python2.  Also avoid installing Anaconda, or python from python.org.  You will then have many different versions of python.  I highly recommended using terminal and Brew to install python3 which is the cleanest way.  If you have already installed Anaconda or python from python.org, uninstall them using any uninstaller tool.  When you type <code>$ python</code> follow by TAB, the TAB should only show python2.7 which is pre-installed.  Do not delete this version or your MacOS will not work.  You can also check <code>$ python3 -V</code> and should return nothing.  You can also try <code>$ which python python3</code> to see which python you are using.  For advanced users who are maintaining lots of python version, you may want to set alias in bash_profile to set which one you are using, or of course, you can set up virtual environments using a particular python version. Phew...

1.  First off, (strangley), MacOS decided to include a Command Line Tool app inside Xcode.  Thus we have to first install Xcode from the App store.  Then, to install the Command Line Tool app, we perform

```
$ xcode-select --install
```

2. Install homebrew.  (this is like apt-get in Linux)

```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

3. Once the installation process is complete, we’ll put the Homebrew directory at the top of the PATH environment variable. This will ensure that Homebrew installations will be called over the tools that Mac OS X may select automatically that could run counter to the development environment we’re creating.

```
$ nano ~/.bash_profile
```
Open the file and write the following:

```
export PATH=/usr/local/bin:$PATH
```

Activate the source
```
source ~/.bash_profile
```

4.  Check whether Homebrew was successfully installed:

```
$ brew doctor
```

5. Install python3

```
$ brew install python3
```

Along with Python 3, Homebrew will install pip, setuptools and wheel.


6.  Try install packages by replacing <package_name> with, e.g., numpy

```
$ pip3 install <package_name>
```


=====================================================================



# <a name="1-virtualEnvironment"></a>2. Virtual Environment
*For beginners, you can ignore this part but we recommend you to try it out for good practice.*

> Virtual Environment is used to manage Python packages for different projects. Using virtualenv allows you to avoid installing Python packages globally which could break system tools or other projects. --- [python.org](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#:~:text=virtualenv%20is%20used%20to%20manage,system%20tools%20or%20other%20projects.)

Remember that a package that we download using `pip3` is written by a human. There may be updates to the package for various reasons. Updating the package may cause your projects to crash due to conflicting in the newer version of the library.

To overcome this problem (and many more reasons to do the following), we will set up a Virtual Environment.

## <a name="2-1-Linux"></a> 1. For Linux User

1. Our environment manager of choice is `venv`
```
$ sudo apt install python3-venv
```
2. I will set up my first environment for this course names pythonDSAI.
```
$ python3 -m venv pythonDSAI
```
3. It should create a directory that names after the environment name.
```
$ ls pythonDSAI/
bin  include  lib  lib64  pyvenv.cfg  share
```
4. To activate the environment, you have to call the `activate` file under your newly created environment folder using `source` command. 
```
$ source pythonDSAI/bin/activate
(pythonDSAI) $ 
```
5. To exit the environment, simply type `deactivate` from anywhere.
```
(pythonDSAI) $ deactivate
$ 
```

## <a name="2-2-windows"></a> 2. For Windows User

1. Download `virtualenv` via `pip install`
```
C:>pip install virtualenv
```
2. I will set up my first environment for this course names pythonDSAI.
```
C:>python -m virtualenv pythonDSAI
```
3. It should create a directory that names after the environment name.
```
C:>dir pythonDSAI
 Volume in drive C has no label.
 Volume Serial Number is 569F-FEDC

 Directory of C:\Users\<username>\pythonDSAI

06/04/2020  11:41 AM    <DIR>          .
06/04/2020  11:41 AM    <DIR>          ..
06/04/2020  11:41 AM                42 .gitignore
06/04/2020  11:41 AM    <DIR>          Lib
06/04/2020  11:41 AM               429 pyvenv.cfg
06/04/2020  11:41 AM    <DIR>          Scripts
               2 File(s)            471 bytes
               4 Dir(s)  42,885,738,496 bytes free
```
4. To activate the environment, you have to call the `activate` file under your newly created environment folder. 
```
C:>pythonDSAI\Scripts\activate
(pythonDSAI) C:>
```
5. To exit the environment, simply type `deactivate` from anywhere.
```
(pythonDSAI) C:>deactivate
C:>
```

## <a name="2-3-macos"></a> 2. For MacOS User

1. First, create a place to place your programming enviornments.  It can be anyname, here my name is Environments

```
$ mkdir Environments
$ cd Environments
```

2. Inside the directory, create enviroment with any name, here I use the name pythonDSAI
```
$ python -m venv pythonDSAI
```

3. To activate
```
$ source pythonDSAI/bin/activate
```

4.  Inside the environment, you can use only python and pip, instead of python3 and pip3 since the programming environment depends on the version that you use to create.  Try verify by
```
$ python -V
$ pip -V
```

5. You probably want to upgrade your pip
```
pip install --upgrade pip
```

6.  You also would be lazy to type source....activate everytime.  Thus, you can open

```
$ nano ~/.bash_profile
```

Type (make sure to change /path/to/)
```
alias dsai="source /path/to/Environments/pythonDSAI/bin/activate"
```

To make sure bash_profile is always sourced when you open terminal, open

```
$ nano ~/.zshrc
```

Type
```
source /path/to/.bash_profile
```

7.  Now try to close and open your terminal again.  Simply type
```
$ dsai
```
It should now activate your env.

================================================================================


## <a name="2-3-action"></a> Virtual Environment in action

We have just installed `numpy` under our global environment. For the best practice, we want our `numpy` to be exist under the environment we wanted.
1. Check that `numpy` is installed.

#### Linux
```
$ pip3 list | grep numpy
numpy         1.18.4 
``` 
*`pip3 list` is a command for listing all the installed packages. `grep` is a command that filters out the text and shows the line that has `numpy` in it. The vertical bar `|` is commonly referred to as a "pipe". It is used to pipe one command into another. That is, it directs the output from the first command into the input for the second command.*


#### Windows
```
C:>pip list | find /I "numpy"
numpy              1.18.4
``` 
*`pip list` is a command for listing all the installed packages. `find /I` is a command that filters out the text and shows the line that has `numpy` in it. The vertical bar `|` is commonly referred to as a "pipe". It is used to pipe one command into another. That is, it directs the output from the first command into the input for the second command.*


#### MacOS
```
$ pip3 list | grep numpy
numpy         1.18.4 
```


2. remove `numpy` from global

#### Linux
```
$ pip3 uninstall numpy
```

#### Windows
```
$ pip uninstall numpy
```

#### MacOS
```
$ pip3 uninstall numpy
```

3. Activate your target environment.

#### Linux
```
$ source pythonDSAI/bin/activate
(pythonDSAI) $
```

#### Windows
```
C:>pythonDSAI\Scripts\activate
(pythonDSAI) C:>
```

#### MacOS
```
$ source pythonDSAI/bin/activate
(pythonDSAI) $
```

4. Install and verify the `numpy` package only exist in the target environment.

#### Linux
```
(pythonDSAI) $ pip3 install numpy
(pythonDSAI) $ pip3 list | grep numpy
numpy         1.18.4 
(pythonDSAI) $ deactivate
$ pip3 list | grep numpy
$
```

#### Windows
```
(pythonDSAI) C:>pip install numpy
(pythonDSAI) C:>pip list | find /I "numpy"
numpy      1.18.4
(pythonDSAI) C:>deactivate
C:>pip list | find /I "numpy"
C:>
```

#### MacOS
```
(pythonDSAI) $ pip3 install numpy
(pythonDSAI) $ pip3 list | grep numpy
numpy         1.18.4 
(pythonDSAI) $ deactivate
$ pip3 list | grep numpy
```

5. Here are the list of library you may need for the next four lectures.
``` 
numpy torch torchvision matplotlib pandas seaborn pandas_datareader sklearn
```
For Linux and Windows user, you will need to install PyTorch using below command
```
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

# <a name="3-jupyter"></a> 3. Jupyter
[Reference](https://jupyter.org/install)

You will need a text editor to write a code. While there are many great python editors out there, we will use Jupyter for this course to simplify the process of checking your works. 

You can install Jupyter using `pip3` (Linux) and `pip` (Windows) in our global python.

1. Install Jupyter via `pip`

#### Linux
```
$ pip3 install jupyter
```
If you receive the following message.

![alt](https://raw.githubusercontent.com/chaklam-silpasuwanchai/Python-for-DS-AI/master/.0%20-%20installation_image/jupyter-path.png)

You have to add `PATH` of the executable scripts.

```
$ echo 'export PATH="~/.local/bin:$PATH"' >> ~/.bashrc
$ source ~/.bashrc
```
#### Windows
```
C:> pip install jupyter
```
#### MacOS

```
$ pip3 install jupyter
```


2. To start jupyter simply type the following command. Your current folder will be the workspace.
```
jupyter notebook
```
![alt](https://raw.githubusercontent.com/chaklam-silpasuwanchai/Python-for-DS-AI/master/.0%20-%20installation_image/jupyter-home.png)

3. Notice that when you try to create a new file there is a `python3` option. This option will run the code using the global environment. Therefore, you will need to add pythonDSAI into the jupyter before we can use it.

![alt](https://raw.githubusercontent.com/chaklam-silpasuwanchai/Python-for-DS-AI/master/.0%20-%20installation_image/jupyter-new.png)  

Activate pythonDSAI environment and add the environment into jupyter

#### Linux
```
$ source pythonDSAI/bin/activate
(pythonDSAI) $ pip3 install ipykernel
(pythonDSAI) $ python3 -m ipykernel install --user --name pythonDSAI --display-name "pythonDSAI"
Installed kernelspec pythonDSAI in /home/<your username>/.local/share/jupyter/kernels/pythondsai
(pythonDSAI) $ deactivate
$
```

#### Windows
```
C:>pythonDSAI\Scripts\activate
(pythonDSAI) C:>pip install ipykernel
(pythonDSAI) C:>python -m ipykernel install --user --name pythonDSAI --display-name "pythonDSAI"
Installed kernelspec pythonDSAI in C:\Users\<username>\AppData\Roaming\jupyter\kernels\pythondsai
(pythonDSAI) C:>deactivate
C:>
```

#### MacOS
```
$ source pythonDSAI/bin/activate
(pythonDSAI) $ pip3 install ipykernel
(pythonDSAI) $ python3 -m ipykernel install --user --name pythonDSAI --display-name "pythonDSAI"
Installed kernelspec pythonDSAI in /home/<your username>/.local/share/jupyter/kernels/pythondsai
(pythonDSAI) $ deactivate
```


4. Restart the jupyter notebook and you should have `pythonDSAI` environment.

![alt](https://raw.githubusercontent.com/chaklam-silpasuwanchai/Python-for-DS-AI/master/.0%20-%20installation_image/jupyter-new-pythonDSAI.png)
