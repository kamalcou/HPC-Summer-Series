# GPU-Accelerated Processing for Machine Learning on UAHPC

Contributors: Dakeya Chambers
University of Alabama Libraries Research Computing Services

Last updated July 10, 2026

In this exercise, we will compare the performance of a data augmentation image processing workflow on both the CPU and the GPU.

Many machine learning workflows rely on large and diverse datasets. However, collecting such datasets can be time-consuming and expensive. One way to address this challenge is through data augmentation.

Data augmentation applies transformations such as rotation, blurring, or scaling to artificially increase the size and variability of a dataset. This helps machine learning models generalize more effectively to new data. 

Image processing involves analyzing and transforming images to extract meaningful information. In computing, an image is typically represented as a two- or three-dimensional array of pixel values.

Because these arrays can be manipulated mathematically, image processing tasks can be framed as computational problems, making them well-suited for optimization using parallel computing.

The dataset used in this exercise is [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master?tab=readme-ov-file), which contains 60,000 training images and 10,000 test images across 10 
clothing categories, including T-shirts, shirts, trousers, pullovers, and other clothing types. For this exercise, the dataset has been artificially expanded to 1,200,000 images. This was done by 
replicating the dataset, rotating each image by 180 degrees, combining the modified with the original, and then multiplying the combined array by a factor of 10.

You will begin by implementing the workflow using NumPy on the CPU, then convert the same workflow to CuPy to run on a GPU. 

This exercise is designed to highlight key concepts in GPU programming and high-performance computing, including data movement and manipulation, as well as efficient use of GPU resources. 

### Learning Outcomes

At the end of this exercise, you should be able to:

- Access UAHPC resources
- Transfer files between your local machine and UAHPC
- Convert NumPy-based workflows to CuPy for GPU execution
- Request and use GPU resources on an HPC system
- Understand how multiple files can be processed within a workflow

## GPU Programming Fundamentals

GPU programming allows computations to be executed in parallel using GPU accelerators. Unlike CPUs, which are optimized for sequential tasks, GPUs contain thousands of smaller cores designed to handle many operations at once. 

One of the most widely used GPU programming platforms is CUDA, which provides a framework for executing code directly on NVIDIA GPUs. CUDA enables developers to leverage GPU hardware for parallel computation.

CuPy is a NumPy-compatible array library for GPU computing in Python. It allows users to write code that closely resembles NumPy workflows while executing operations on a GPU instead of a CPU.
Because of this compatibility, existing NumPy-based code can often be converted to CuPy with minimal changes, making it an accessible way to introduce GPU acceleration into data processing workflows.

For more information, see the official [CuPy documentation](https://docs.cupy.dev/en/stable/overview.html).


## Required Files

You should have access to the MNIST Fashion dataset through UA Box. Navigate to the **data** folder and download all of the contents locally to your Downloads folder and extract them. 

## Accessing UAHPC

1. Open a terminal or Windows Powershell. You will need to connect to the UAHPC server using this command:

```bash
ssh {myBamaID}@uahpc.ua.edu
```

You will be prompted to complete Duo two-factor authentication and enter your myBama password. 

We need to upload our images and scripts to UAHPC. We will first use `pwd` to see what directories we currently have and list the files:
```bash
[myBamaID@uahpc-login001 ~]$ pwd
[myBamaID@uahpc-login001 ~]$ ls
```
Next, make a directory on HPC to hold our data files:
```bash
[myBamaID@uahpc-login001 ~]$ mkdir data
```
## SFTP

2. Open a second terminal and connect to the SFTP server using the following command: 

```bash
sftp> {myBamaID}@uahpc.ua.edu
```
You will then be prompted to enter your password, and complete Duo two-factor authentication. After successfully connecting to the server you should see a system prompt like this:

```sftp
sftp>
```

First, print the remote working directory:

```sftp
sftp> pwd
```

Copy the entire contents of the downloaded folder by using this command: 
```sftp
put -r C:\Users\myBamaID\Downloads\data/*
```
Confirm the transfer was successful:
```bash
cd data
ls
```

You should have **5** files in total:

`mnist_fashion.npy`, `cupy_pipeline.py`, `cupy_pipeline.batch`, `numpy_pipeline.py`, and `numpy_pipeline.batch`


## Data Augmentation Pipeline 

In this section, you will run two versions of the same data augmentation pipeline:

- A NumPy based version that runs on the CPU
- A CuPy based version that runs on the GPU

Both scripts perform the same image transformations, but they differ in how the computations are executed. The goal is to 
compare performance and understand how the data is processed on different hardware. 


### Creating the Conda Environment
Before running our scripts, we will need to create a conda environment that will be able to run both scripts. Follow the steps below:

```bash
[myBamaID@uahpc-login001 data]$ module load miniconda3/base

[myBamaID@uahpc-login001 data]$ conda create --name imgproc -y 

[myBamaID@uahpc-login001 data]$ conda activate imgproc

(imgproc)[myBamaID@uahpc-login001 data]$ conda install -c conda-forge python numpy cupy scipy matplotlib scikit-image -y

(imgproc)[myBamaID@uahpc-login001 data]$ conda deactivate

```

### NumPy Image Preprocessing

This script implements the pipeline using NumPy and standard Python libraries. All computations are performed on the CPU sequentially. This approach can become inefficient for large datasets due to the lack of parallelism and slower processing speed. In short, this script:

- Loads a batch of images
- Converts image data into array format
- Applies three image transformations: blur, sharpening, and gamma correction
- Saves the processed datasets
- Measures total runtime on the CPU

We will first run this script "as is" and use it as a baseline. 

To do this, we first need to adjust the `DATA_PATH` line in the **numpy_pipeline.py** to replace "myBamaID" to your myBama username. To do this, we use `nano` to open and edit the Python script.
```bash
nano numpy_pipeline.py
```
Make the edits; then use `CTRL+O` to save and `CTRL+X` to exit.

#### Python script: numpy_pipeline.py

```python
# Import libraries
import time 
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import unsharp_mask
import os

# This script developed with assistance from Github Copilot

# -------- Student Inputs (edit these) --------
DATA_PATH = "/home/myBamaID/data/mnist_fashion.npy"      # Replace this with the path to your dataset
OUTPUT_DIR = "numpy_processed_output"         

GAUSS_SIGMA = 1.0      # Try: 0.5, 2.0
SHARP_RADIUS = 1.0     # Try: 0.5, 2.0
SHARP_AMOUNT = 2.0     # Try: 1.0, 3.0
GAMMA = 0.5            # Try: 0.2, 1.0, 1.5
NUM_SAMPLES = 8        # Number of images to preview
# --------------------------------------------



def load_mnist_images(file_path):
    #Returns array of shape (num_images, height, width)
    data = np.load(file_path)
    return data



def preprocess_batch(file_path, save_dir=OUTPUT_DIR):
    
    os.makedirs(save_dir, exist_ok=True)

    # Load raw MNIST image data. These are grayscale images stored in binary format
    images = load_mnist_images(file_path)

    # Convert pixel values from [0, 255] integers to [0,1] floats
    sample_images = images.astype(np.float32) / 255.0
    print(f"Loaded {len(images)} images.")

    # Gaussian Blur to smooth the image
    gauss_blur = gaussian_filter(sample_images, sigma=(0, GAUSS_SIGMA, GAUSS_SIGMA))

    # Sharpening enhances the image's edges and details
    sharpened = unsharp_mask(
        sample_images,
        radius=SHARP_RADIUS,
        amount=SHARP_AMOUNT
    )

    # Gamma correction adjusts the brightness and contrasts of the images
    gamma_correction = sample_images ** GAMMA


    # Save processed results as numpy arrays 
    np.save(f"{save_dir}/gauss_blur.npy", gauss_blur)
    np.save(f"{save_dir}/gamma_correction.npy", gamma_correction)
    np.save(f"{save_dir}/sharpened.npy", sharpened)

    return sample_images


# Run the pipeline and calculate total compute time
start = time.process_time()

processed = preprocess_batch(DATA_PATH)

end = time.process_time()
print(f"CPU runtime: {end - start:.4f} seconds")


# Load processed data 
gauss_blur = np.load(f"{OUTPUT_DIR}/gauss_blur.npy")
gamma_correction = np.load(f"{OUTPUT_DIR}/gamma_correction.npy")
sharpened = np.load(f"{OUTPUT_DIR}/sharpened.npy")


# Visualization - selects 8 random images from the dataset

rng = np.random.default_rng()
indices = rng.integers(0, len(gauss_blur), size=NUM_SAMPLES)

fig, axes = plt.subplots(3, NUM_SAMPLES, figsize=(20, 8))

row_labels = ["Gaussian Blur", "Sharpened", "Gamma Correction"]
arrays = [gauss_blur, sharpened, gamma_correction]


for row, (label, arr) in enumerate(zip(row_labels, arrays)):
    for col, idx in enumerate(indices):
        axes[row, col].imshow(arr[idx], cmap='gray')
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)

plt.suptitle("NumPy Pipeline (CPU)")
plt.tight_layout()
plt.savefig("mnist_np_pipeline_sample.png")
```

### Running the Batch Script

#### Batch Script
```bash
#!/bin/bash
#SBATCH --job-name=python_cpu
#SBATCH --output=output_cpu.%A
#SBATCH --error=error_cpu.%A
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G 
#SBATCH -p main
#SBATCH -q main
#SBATCH --time=30:00

# Load Conda module
module load miniconda3/base

# Activate the environment
conda activate imgproc

# Run the script
python numpy_pipeline.py
```

To run the python file, submit the following batch script using: 
```bash
sbatch numpy_pipeline.batch
```
To check on the status of a running job, use the following:
```bash
squeue --me
```

You can view the outputs using this command:
```bash
cat output_cpu.jobid
```

To view the visualization outputs, we will need to transfer files back to our local computer. Return to the second terminal and navigate to the remote directory:
```sftp
cd /home/myBamaID/data
```

Use `get` with wildcard to copy the .png file:
```
get *.png 
```

### Exercises:

After testing the first script, now modify any of the following parameters one by one and observe the results. Alternative suggested values are in the script. 

- Change `GAUSS_SIGMA` to increase or decrease Gaussian Blur (Try: 0.5 or 2.0)
- Adjust `GAMMA` to brighten the images (Try 0.2, 1.0, and 1.5)
- Increase `NUM_SAMPLES` and see how it affects runtime

#### Questions to Answer
- How does changing parameters affect runtime?
- Which operations seem most computationally expensive?



### CuPy Image Preprocessing

GPUs are designed for parallel execution so they can significantly accelerate operations that are applied repeatedly across large datasets. In the CuPy version, parallelism is handled automatically
by the GPU. When data is converted to a CuPy array using `cp.asarray()`, computations are offloaded to the GPU. 

One thing to note is that we do not explicitly manage parallelization in our code; instead, CuPy uses underlying CUDA kernels and optimized GPU libraries to distribute the work across thousands 
of GPU threads. You can think of CuPy as providing the pre-built GPU functions. For example, we did not write the implementation of `gaussian_filter()` ourselves, nor did we write the CUDA kernels that perform the parallel computation. 
We call the function, and CuPy handles the GPU execution behind the scenes. This is what makes the CuPy library efficient and easy to use. 

In this script:
- Arrays are stored on the GPU using `cp.asarray()` instead of NumPy arrays
- Computations are executed in parallel across GPU cores
- Data must be transferred between CPU (host) and GPU (device)
- Runtime is measured using GPU timing methods

First, we will need to verify GPU access. Use `nano` to create a file named `gpu_test.batch`

```bash
nano gpu_test.batch
```

Copy the following code into the file:

``` bash
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test.%A
#SBATCH --error=gpu_error.%A
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00

nvidia-smi
```
Use `CTRL+O`, and `enter` to save and `CTRL+X` to exit.

We will now run the test batch script:
```bash
sbatch gpu_test.batch
```
The batch script above requests a single GPU node and runs the `nvidia-smi` command to verify that the GPU is available and that the NVIDIA drivers are loaded correctly. 
The output displays information about the GPU hardware, including the GPU model, driver version, CUDA version, memory usage, and currrent utilization. 
To view the output use:
```bash
cat gpu_test.%A
```

Now, we will first run the following script "as is" and use it as a baseline. Adjust the `DATA_PATH` line in the **cupy_pipeline.py** to replace "myBamaID" to your myBama username. 

```bash
nano cupy_pipeline.py
```

### Python script: cupy_pipeline.py

```python
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import convolve, gaussian_filter
import os
import matplotlib.pyplot as plt

DATA_PATH = "/home/myBamaID/data/mnist_fashion.npy"
OUTPUT_DIR = "cupy_processed_output"
GAUSS_SIGMA = 1.0
SHARP_STRENGTH = 1.0
GAMMA = 0.5
NUM_SAMPLES = 8

def preprocess_mnist(file_path, save_dir=OUTPUT_DIR, batch_size=1000, num_streams=4):
    os.makedirs(save_dir, exist_ok=True)
    images = np.load(file_path)
    num_batches = -(-len(images) // batch_size)

    print(f"Processing {len(images)} images in {num_batches} batches with {num_streams} streams")

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    kernel = cp.array([[0, -1, 0], [-1, 5+SHARP_STRENGTH, -1], [0, -1, 0]],
                      dtype=cp.float32).reshape(1, 3, 3, 1)

    results = {"gauss_blur": [], "sharpened": [], "gamma_correction": []}

    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    start_event.record()

    for batch_idx in range(num_batches):
        with streams[batch_idx % num_streams]:
            batch = images[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            sample_images = cp.asarray(batch[:, :, :, np.newaxis]).astype(cp.float32) / 255.0

            results["gauss_blur"].append(cp.asnumpy(
                gaussian_filter(sample_images, sigma=(0, GAUSS_SIGMA, GAUSS_SIGMA, 0))))
            results["sharpened"].append(cp.asnumpy(
                cp.clip(convolve(sample_images, kernel), 0, 1)))
            results["gamma_correction"].append(cp.asnumpy(sample_images ** GAMMA))


    for stream in streams:
        stream.synchronize()

    end_event.record()
    end_event.synchronize()
    print(f"GPU runtime: {cp.cuda.get_elapsed_time(start_event, end_event):.2f} ms")

    for name, data_list in results.items():
        np.save(f"{save_dir}/{name}.npy", np.concatenate(data_list, axis=0))

    print(f"Results saved to {save_dir}/")

# Run pipeline
preprocess_mnist(DATA_PATH)

# Visualization
gauss_blur = np.load(f"{OUTPUT_DIR}/gauss_blur.npy")
gamma_correction = np.load(f"{OUTPUT_DIR}/gamma_correction.npy")
sharpened = np.load(f"{OUTPUT_DIR}/sharpened.npy")

rng = np.random.default_rng()
indices = rng.integers(0, len(gauss_blur), size=NUM_SAMPLES)

fig, axes = plt.subplots(3, NUM_SAMPLES, figsize=(20, 8))
for row, (label, arr) in enumerate(zip(["Gaussian Blur", "Sharpened", "Gamma Correction"],
                                        [gauss_blur, sharpened, gamma_correction])):
    for col, idx in enumerate(indices):
        axes[row, col].imshow(arr[idx, :, :, 0], cmap='gray')
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)

plt.suptitle("CuPy Pipeline (GPU)")
plt.tight_layout()
plt.savefig("mnist_cupy_pipeline_sample.png")
```
### Running the Batch Script

#### Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=python_gpu
#SBATCH --output=output_gpu.%A
#SBATCH --error=error_gpu.%A
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00

# Load the Conda module
module load miniconda3/base

# Activate the environment
conda activate imgproc

# Run the script
python cupy_pipeline.py
```

To run the python file, submit the following batch script using: 
```bash
sbatch cupy_pipeline.batch
```

To check on the status of a running job, use the following:
```bash
squeue --me
```
You can view the outputs using this command:
```bash
cat output_gpu.jobid
```
To view the visualization outputs, we will need to transfer files back to our local computer. Return to the second terminal and navigate to the remote directory:
```sftp
cd /home/myBamaID/data
```

Use `get` with wildcard to copy multiple files:
```
get *.png 
```

### Exercises:

After testing the first script, now modify any of the following parameters one by one and observe the results. Alternative suggested values are in the script. 

- Increase `NUM_SAMPLES` observe if it affects runtime
- Adjust `GAMMA` to brightnen or darken images (Try: 0.2, 1.0, 1.5)
- Compare GPU runtime to the CPU version for the same parameters


#### Questions to Answer
- How does GPU performance compare to CPU performance for this workflow?
- What role does data transfer between CPU and GPU play in overall performance?
