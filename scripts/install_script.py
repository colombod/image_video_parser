# Cuda Installation script

import wget
import shutil

def install_cuda():
    print("Installing Cuda")
    os = platform.system()
    if os == "Windows":
        install_cuda_for_windows()
    elif os == "Linux":
        install_cuda_for_2204()
    else:
        print("OS not supported")
        exit(1)

def install_cuda_for_windows():
    print ("hold")
# get url for cuda for windows
# get url for cudo installer for windows 

def install_cuda_for_2204():
    cuda_pin_url = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin"
    cuda_pin_filename = wget.download(cuda_pin_url)

    cuda_repo_url = "https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-ubuntu2204-12-5-local_12.5.1-555.42.06-1_amd64.deb"
    cuda_repo_filename = wget.download(cuda_repo_url)

    #Move cuda files
    shutil.move('cuda-ubuntu2204.pin' '/etc/apt/preferences.d/cuda-repository-pin-600')

  

    
