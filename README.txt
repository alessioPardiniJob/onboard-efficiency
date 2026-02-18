## Prerequisites

To run this project, your system must meet the following general requirements. 

* **Operating System:** Linux (Ubuntu, Debian, etc.) or Windows via WSL2.
* **Python:** Version >= 3.8, including the `venv` module (e.g., `sudo apt install python3-venv`).
* **System Tools:** `make`, `build-essential`, and `git`.
* **Network:** An active internet connection to download Python dependencies.

If your environment already meets these requirements, you can proceed directly to running the `Makefile`. However, to ensure maximum compatibility and avoid altering your system-level Python configuration, **we highly recommend using the isolated Pyenv setup detailed below.**

---

## Recommended Quick Setup (Pyenv)

This guide uses `pyenv` to install Python 3.10 safely into the user space, ensuring a reproducible environment without affecting your OS package manager.

### 1. Install System Dependencies
Ensure you have all the necessary build tools and libraries required to compile Python:

```bash
sudo apt update
sudo apt install -y build-essential make git cmake \
libbz2-dev libssl-dev libffi-dev libncurses5-dev libncursesw5-dev \
libreadline-dev libsqlite3-dev liblzma-dev zlib1g-dev tk-dev xz-utils \
wget curl llvm
2. Install Pyenv
If pyenv is not already installed on your system, install it and configure your shell environment:

Bash
curl [https://pyenv.run](https://pyenv.run) | bash

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

source ~/.bashrc
3. Project Configuration
Navigate to the root directory of the cloned repository and install the required Python version:

Bash
# Replace with the actual path where you cloned the repository
cd path/to/onboard-efficiency

# Install Python 3.10.13 and set it locally for this directory
pyenv install 3.10.13
pyenv local 3.10.13
4. Setup Project Environment
Use the provided Make target to automatically create the virtual environment and install all project dependencies:

Bash
make setup-EuroSAT
5. Verify Installation
Verify that the correct Python version is active and the setup was successful:

Bash
python --version  # Expected output: Python 3.10.13
make --version

# Activate the virtual environment to start working
source EuroSAT/venv/bin/activate
✅ All dependencies are now installed. The project is ready to run.
