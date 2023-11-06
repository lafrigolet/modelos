

**Step 1: Update Package List**

Before installing any software, it's a good practice to update your package list to ensure you have the latest package information. Open a terminal and run the following command:

```bash
sudo apt update
```

**Step 2: Install Python**

Python is often pre-installed on Debian-based systems, but you can ensure you have the latest version by running:

```bash
sudo apt install python3
```

This command installs Python 3, which is recommended for most new projects, as Python 2 has reached its end of life.

**Step 3: Install Pip (Python Package Manager)**

Pip is a package manager for Python. You can install it with the following command:

```bash
sudo apt install python3-pip
```

This installs Pip for Python 3.

**Step 4: Verify Python and Pip Installation**

You can check the versions of Python and Pip that you've just installed:

```bash
python3 --version
pip3 --version
```

This should display the version information for Python 3 and Pip 3.

**Step 5: Create a Virtual Environment:**
Create a virtual environment with the following command. Replace myenv with the name you prefer for your virtual environment:

bash
Copy code
python3 -m venv myenv

**Step 6: Activate the Virtual Environment:**
Activate your virtual environment with the following command:

bash
Copy code
source myenv/bin/activate
Your command prompt should change, indicating that you are now in the virtual environment.

**Step 7: Install PyTorch**

You can install PyTorch, a popular deep learning framework, using Pip. There are various installation options available for PyTorch, depending on your system and requirements. Here's a basic example of how to install the CPU-only version of PyTorch:

```bash
pip3 install torch torchvision torchaudio
```

This command installs PyTorch, its computer vision library torchvision, and audio library torchaudio.

**Step 8: Verify PyTorch Installation**

You can verify the PyTorch installation by opening a Python interpreter:

```bash
python3
```

Then, in the Python interpreter, you can import PyTorch:

```python
import torch
print(torch.__version__)
```

This should display the PyTorch version you've installed.

**Step 9: Verify PyTorch is running on GPU**

Certainly! You can use a simple PyTorch program to check if it's running on a GPU. Here's a Python script that checks for the availability of a GPU and prints the device name:

```python
import torch

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Create a tensor on the selected device
x = torch.rand(3, 3).to(device)

# Print the device of the tensor
print("Tensor is on", x.device)
```

Save this script to a `.py` file and run it. If a GPU is available, it will print the GPU's name, and if not, it will indicate that it's using the CPU.

Make sure you have PyTorch installed and that it's configured to work with your GPU. If you have a compatible NVIDIA GPU and have installed the GPU version of PyTorch, this script should detect and use the GPU for tensor operations.
