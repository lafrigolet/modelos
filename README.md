# modelos
## Requirements
OS: Debian 12
Python: 3.9
Python Requirements:
certifi==2023.7.22
charset-normalizer==3.3.1
cmake==3.27.7
filelock==3.13.1
idna==3.4
Jinja2==3.1.2
lit==17.0.4
MarkupSafe==2.1.3
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.1
Pillow==10.1.0
requests==2.31.0
sympy==1.12
torch==2.0.1+cu117
torchaudio==2.0.2+rocm5.4.2
torchvision==0.15.2+rocm5.4.2
triton==2.0.0
typing_extensions==4.8.0
urllib3==2.0.7
## Python 3.9 build on Debian 12
To install Python 3.9 on Debian, you can follow these steps. Debian typically comes with Python 3 pre-installed, but you can install Python 3.9 alongside it.

1. **Update Package Lists**:

   Open a terminal and run the following command to update the package lists and upgrade any existing packages to their latest versions:

   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **Install Required Dependencies**:

   Before building Python from source, you'll need to install some development tools and libraries. Run the following command to install the necessary dependencies:

   ```bash
   sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev
   ```

3. **Download and Compile Python 3.9**:

   Next, download the Python 3.9 source code, extract it, and compile it. You can use the following commands:

   ```bash
   # Create a directory for the Python source code
   mkdir ~/python39

   # Navigate to the new directory
   cd ~/python39

   # Download Python 3.9 source code
   wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz

   # Extract the source code
   tar -xvf Python-3.9.7.tgz

   # Navigate into the Python source directory
   cd Python-3.9.7
   ```

   Now you're in the Python 3.9 source directory.

4. **Configure and Compile Python**:

   Run the following commands to configure and compile Python 3.9:

   ```bash
   ./configure --enable-optimizations
   make -j8  # This will use 8 CPU cores for faster compilation; adjust it as needed
   ```

   The `--enable-optimizations` flag enables various optimizations during compilation.

5. **Install Python**:

   After the compilation is complete, install Python 3.9:

   ```bash
   sudo make altinstall
   ```

   Using `altinstall` is preferred over `make install` because it installs Python 3.9 alongside the system Python without replacing it.

6. **Verify Python Installation**:

   You can verify that Python 3.9 is installed correctly by running:

   ```bash
   python3.9 --version
   ```

   You should see the Python 3.9 version information.

Python 3.9 is now installed on your Debian system. You can use it by running `python3.9` or `python3.9 <script.py>` for your Python scripts.
