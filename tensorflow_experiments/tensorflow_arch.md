Installing TensorFlow with python3-support on archlinux
-------------------------------------------------------

To install TensorFlow on archlinux (or any any other up to date distro I suppose) do the following:

1. Find the link to the python3 version to download from [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
2. Rename the file so that `cp34` which corresponds "python version 3.4" now reflects the actually installed version, which in my case is `cp35` as I have version 3.5.1 (check with `python --version`, but do not include the patch-version number, just the major and minor version)
3. Install the file with pip3

In the shell

```bash
# Download the file to the current folder
wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp34-none-linux_x86_64.whl
# Rename it to reflect the actual python version (here 3.4 to 3.5)
mv tensorflow-0.6.0-cp34-none-linux_x86_64.whl tensorflow-0.6.0-cp35-none-linux_x86_64.whl
# Install it with pip3
sudo pip3 install --upgrade tensorflow-0.6.0-cp35-none-linux_x86_64.whl
```

This can of course be used together with virtualenv to avoid the possibility of an upset pacman...

Just create and enter the environment first:

```bash
# Create the virtual environment
virtualenv --system-site-packages ~/tensorflow
# "Enter" the environment
source ~/tensorflow/bin/activate
```

Then issue the commands described earlier
