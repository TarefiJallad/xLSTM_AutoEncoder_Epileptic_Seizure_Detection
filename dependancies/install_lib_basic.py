import subprocess
import sys

def install_packages(packages):
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.\n")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}.\n")

if __name__ == "__main__":
    required_packages = [
        "mne",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "joblib",
        "h5py"
    ]
    install_packages(required_packages)
