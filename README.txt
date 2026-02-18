## Prerequisites

- **OS:** Linux (Ubuntu, Debian, etc.) or Windows with WSL2  
- **Python:** ≥3.8 with `venv` module (`sudo apt install python3-venv`)  
- **System tools:** `make`, `build-essential`, `git`  
- **Internet connection** to install Python dependencies  

### Verify installation
```bash
python3 --version
pip3 --version
make --version


# Quick Setup – Python 3.10 (Pyenv)

## Prerequisites
```bash
sudo apt update
sudo apt install -y build-essential make git cmake \
libbz2-dev libssl-dev libffi-dev libncurses5-dev libncursesw5-dev \
libreadline-dev libsqlite3-dev liblzma-dev zlib1g-dev tk-dev xz-utils \
wget curl llvm
Pyenv & Python 3.10
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.10.13
cd ~/onboard-efficiency
pyenv local 3.10.13
Project Setup
make setup-EuroSAT
Verify
python --version
pip --version
make --version
Activate venv (optional)
source EuroSAT/venv/bin/activate
✅ All dependencies installed. Ready to run the project.
