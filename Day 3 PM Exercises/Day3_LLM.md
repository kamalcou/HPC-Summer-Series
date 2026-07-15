# Running Large Language Models (LLMs) on a HPC Cluster

Contributors: Cyrus Gomes, Avery Fernandez, Dakeya Chambers, Jacob Hoernlein, and Rachel DuBose
University of Alabama Libraries Research Computing Services

Last updated July 10, 2026


## Overview

In this tutorial, participants will learn about the following:

1. How to connect to the HPC cluster.
2. How to build and install the `llama-cpp` library optimized for the cluster's GPU architecture using a container.
3. How to interactively request GPU resources and use the quantized Phi-4 model to run inference.
4. How to use Slurm batch scripts to run the container and obtain inference from text prompts.
5. How to access OpenAI's gpt-oss-20b and gpt-oss-120b models and compare model outputs.

## Llama-cpp
**Llama-cpp** is fast and lightweight version of C/C++ implementation within the LLaMA‑family language models. **Llama.cpp** strives to enable LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware. It is free, open-source, MIT licensed. Many different LLMs are available to use with **Llama-cpp**. 

For further reading see: [**Llama-cpp Github**](https://github.com/ggml-org/llama.cpp).

We will be using container image of Llama-cpp built for CUDA v.12.4.1 as that was the most recent version of CUDA at the time of development. Please see [here](https://hub.docker.com/r/nvidia/cuda) for license information.  

## Apptainer
**Apptainer** is container software built specifically to run on HPC clusters and other environments where users do not have root access. There is an **Apptainer** module on the HPC which will be loaded. Then, we will use an **Apptainer** Definition file (`.def` file) to build the **Apptainer** container image (`.sif` file). The container image can be executed using **Apptainer** commands as a stand-alone operating system. The instructions for the `.sif` file are provided by the `.def` file.

For more information, visit [**Apptainer**](https://apptainer.org/docs/user/latest/introduction.html) website.

## Model Details
In this tutorial, we are setting up and running the Phi-4 language model (LLM) on a high-performance computing (HPC) cluster using the `llama-cpp` library. The Phi-4 model, released by Microsoft in 2024, is a smaller language model that can handle language processing and complex reasoning tasks. For more information, read the [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905).
- **[Phi-4-Q6.gguf](https://huggingface.co/microsoft/phi-4-gguf)**  
- [**Microsoft AI Considerations**](https://huggingface.co/microsoft/phi-4-gguf#responsible-ai-considerations)  
- [**Microsoft Code of Conduct**](https://huggingface.co/microsoft/phi-4-gguf/blob/main/CODE_OF_CONDUCT.md)

### Common LLM File Types
For this tutorial, we will use Microsoft’s Phi-4 GGUF model, but we have included some of the various model types you may encounter below:

1. **GGML / GGUF (`.ggml`, `.gguf`)**  
   - *Use:* Quantized formats (originally for `llama.cpp`) that greatly reduce model size. Ideal for local inference, including CPU-only setups.  
   - *What it is:* Compressed and quantized model files optimized for GPU inference (and often CPU as well).

2. **bf16 / f16 (`.bf16`, `.f16`)**  
   - *Use:* Intermediate formats for storing model weights in reduced precision (bfloat16 or float16).  
   - *What it is:* Lower-precision floating point formats that reduce memory usage and speed up inference, commonly used before quantization or for efficient GPU inference.

## Prerequisites

- **Terminal/PowerShell**
- **Stable internet connection**
- **HPC account**
- **space on HPC account**

 Note: This tutorial was tested using the terminal/PowerShell.

# Step 1: Connecting to UAHPC

To ssh into the server, open a terminal window or command prompt and type the following command:
```
ssh <myBamaID>@uahpc.ua.edu
```
Enter your myBama password when prompted and follow the Duo multi-factor authentication prompts.

# Step 2: Make directories
Make your LLM directory using:
```bash
mkdir llm
#check to see directory exists
ls
#navigate into llm directory folder
cd llm
```

# Step 3: Apptainer Definition file

Below is the `.def` file that defines the container image. The `.def` file has two main parts: the header and the sections. The header describes the core operating system to build within the container. Then there are several sections including the labels, environment, post, and runscript sections.

- **`%labels`** Section: can be used to add metadata to the file.

- **`%post`** Section: used to download files from the internet with tools like git and wget, install new software and libraries, write configuration files and create new directories.

- **`%environment`** Section: used to define environment variables that will be set at runtime; they are not available during buildtime. 

- **`%runscript`** Section: contents are written to a file within the container that is executed when the container image is run via `apptainer run` command.

There can be other sections such as `%arguments`, `%setup`,`%files`,`%test`, `%startscript`, and `%help` but it is not necessary to include every section (or any sections) in the `.def` file and we will not be using these sections for this tutorial. Note, the order of the sections in the `.def` file does not matter.  

We will use`nano` to create the file.
```bash
nano llama_cuda.def
```
Once, the nano window has opened. Copy and paste the lines below starting at `Bootstrap`.

```linux
Bootstrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04

%labels
    Author R DuBose
    Description GPU-enabled llama.cpp container for HPC

%environment
    export LLAMA_MODEL=/opt/models/phi-4-Q6_K.gguf
    export PATH=/usr/local/bin:$PATH

%post
    set -eux 

    export DEBIAN_FRONTEND=noninteractive

    # Update and install dependencies
    apt-get update 
    apt-get install -y \
        build-essential \
        ca-certificates \
        cmake \
        git \
        pkg-config \
        libssl-dev \
        python3 \
        python3-pip \
        wget
    rm -rf /var/lib/apt/lists/*

    # Make sure CUDA compiler is available
    echo "Verifying CUDA compiler..."
    nvcc --version

    # Preparing source tree
    install -d /opt/src
    rm -rf /opt/src/llama.cpp
    
    echo "Cloning llama.cpp..."
    # Clone llama.cpp
    git clone https://github.com/ggml-org/llama.cpp.git /opt/src/llama.cpp

    echo "verifying clone..."
    ls -la /opt/src
    ls -la /opt/src/llama.cpp || true
    
    test -d /opt/src/llama.cpp
    test -f /opt/src/llama.cpp/CMakeLists.txt || (echo "ERROR: missing CMakeLists.txt; repo not cloned correctly"; exit 1)

    cd /opt/src/llama.cpp
    
    # Ensuring CUDA driver stub is linkable
    STUB_LIB="$(find /usr/local/cuda /usr -maxdepth 12 -type f -name 'libcuda.so' -print -quit 2>/dev/null || true)"
    if [ -n "${STUB_LIB}" ]; then
        STUB_DIR="$(dirname "${STUB_LIB}")"
        if [ ! -e "${STUB_DIR}/libcuda.so.1" ]; then
            ln -sf "${STUB_LIB}" "${STUB_DIR}/libcuda.so.1"
        fi
        export LDFLAGS="-L${STUB_DIR} -Wl,-rpath-link,${STUB_DIR} ${LDFLAGS:-}"
        export LIBRARY_PATH="${STUB_DIR}:${LIBRARY_PATH:-}"
        echo "    Using stub: ${STUB_LIB}"
    else
        echo "    NOTE: libcuda.so stub not found in image. Build may still succeed depending on toolchain."
    fi

    # Build llama.cpp with CUDA
    echo "Configuring CMake..."
    rm -rf build CMakeCache.txt CMakeFiles
    cmake -S . -B build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_COMMON=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}"

    echo "Building..."
    cmake --build build --config Release -j16
    
    echo "Installing..."
    # we will not be using LLAMA-server but it is needed for the build.
    install -m 0755 -D build/bin/llama-cli     /usr/local/bin/llama-cli
    install -m 0755 -D build/bin/llama-server  /usr/local/bin/llama-server
    install -m 0755 -D build/bin/llama-bench   /usr/local/bin/llama-bench || true

    echo "Build complete...sanity checks..."
    test -x /usr/local/bin/llama-cli
    /usr/local/bin/llama-cli --version || true

%runscript
    echo "Running llama.cpp with GPU support..."
    exec /usr/local/bin/llama-cli -m "${LLAMA_MODEL}" "$@"
```

# Step 4. Building Llama-cpp
Now you are ready to build the container.

First, request an interactive compute node to build the container:

```bash
### cpus per task should match what we put in the cmake build command line
srun --pty --partition=main --qos=main --mem=32G --cpus-per-task=16 --ntasks=1 -t 02:00:00 bash
```
We also need to set up some directories for APPTAINER to prevent running out of storage during the build.
```bash
export SCRATCH=/scratch/$USER
export APPTAINER_TMPDIR="$TMPDIR/apptainer-$USER"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache"

mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
```

Next, load the Apptainer module:

```bash
module load apptainer
```

Finally, build the container. We will use the fakeroot feature to run the container. It allows an unprivileged user to run a container with the appearance of running as root. For further information, see [here](https://apptainer.org/docs/user/main/fakeroot.html).
```bash
apptainer build --fakeroot llama_cuda.sif llama_cuda.def
```

# Step 5. Accessing the model 
We made the Phi-4 model available to you at /grps2/ualib/llm. We will access the model from this location.

Now type `exit` to get off the compute node so we can transition to the GPU node.

# Step 6. Running LLM Inference with `llama-cli`
## Interactively requesting GPU resources

### Split into groups of 3-4 to reduce resource demand

Now that the container image has been built, we can request a GPU node interactively. Once we are on the GPU node, we can execute the container and interactively prompt the model.

To request a GPU node, run:
```bash
srun --pty --nodes=1 --gres=gpu:1 --partition=gpu --qos=gpu --cpus-per-task=4 --mem=16G --time=00:30:00 bash
```
Now to run the container, use:
```bash
module load apptainer
apptainer exec --nv -B /grps2:/grps2 $HOME/llm/llama_cuda.sif \
  llama-cli -m /grps2/ualib/llm/phi-4-Q6_K.gguf -ngl 999 -p "Hello from the HPC"
```
Next, try some of the following prompts:
1. List five good study habits.
2. Briefly explain some methods researchers use to analyze large datasets.
3. Generate six short icebreaker questions for an introductory sociology course.
4. List 3 strategies for breaking down complex concepts.
5. State one key reason why peer-review is essential in academic research.

To exit the interactive session, press CTRL+C.

## Running inference with SLURM
This batch script requests one GPU and runs `llama-cli` to generate text using a local GGUF model. The script loads the model file, uses the CPU cores assigned by Slurm, offloads as many model layers as possible to the GPU, and generates up to 512 tokens in response to a prompt.

To run this script (inference.batch), we will also need to `exit` the interactive GPU node by typing `exit`.

### Model Parameters
Here's a table of common parameters and their usage:

Argument | Usage |
|--------|-------|
-h, --help, --usage | print usage and exit|
--version | show version and build information|
-t, --threads| number of threads|
-n, --predict, --n-predict | number of tokens to predict|
-c, --ctx-size | size of the prompt context|
-ngl, --gpu-layers | maximum number of layers to store in VRAM|
-p, --prompt | prompt to begin generating responses
-f, --file | a file containing the prompt|
-st, --single-turn | run conversation for only a single turn|

This information was adapted from the [Llama-cpp Github](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md).

**Inference.batch**
```bash
#!/bin/bash
#SBATCH --job-name=llama_infer
#SBATCH --output=output_llama_infer.%A
#SBATCH --error=error_llama_infer.%A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:1


set -euo pipefail

module load apptainer

########################
# Paths
########################

# Path to your container
CONTAINER="$HOME/llm/llama_cuda.sif"  

########################
# Run
########################
time srun apptainer exec --nv -B /grps2:/grps2 "$CONTAINER" \
    llama-cli \
        -m  /grps2/ualib/llm/phi-4-Q6_K.gguf \
        -n 512 \
        --threads "$SLURM_CPUS_PER_TASK" \
        --ctx-size 0 \
        -ngl 999 \
        --single-turn \
        -p "A mobile app suddenly hits API rate limits after a new feature rollout. Identify likely coding or architectural causes. Explain how you would instrument the system to diagnose the issue. Suggest two mitigation strategies. Present your answer in a table with "Cause" and "Fix"."
```


To create the batch file, use `nano`. 
```bash
nano inference.batch
```
This will open another window; copy the script above into the `nano` window. Hit CTRL+O to save the file, enter, then CTRL+X to exit the `nano` window.

To run `inference.batch`, type `sbatch inference.batch`

To check status of submitted job:
```bash
squeue -u {myBamaID}
```

To view output use the following and replace `%A` with the job number:
```bash
cat output_llama_infer.%A
```
Note that the time to run the model is listed at the bottom of the output file.
To view GPU usage statistics, type `cat error_llama_infer.%A` where `%A` is the job number. This will open the error file for viewing.
## Inline prompting with -p!
We can directly change the prompt for the model by adjusting the argument `-p` or `--prompt`. Use `nano` to change out the text prompt. Let's ask the model to give us some travel tips:

```txt
Plan an affordable weekend trip within driving distance from Tuscaloosa, Alabama.
```
```bash
nano inference.batch
```
This will open another window; copy the script above into the `nano` window. Hit CTRL+O to save the file, enter, then CTRL+X to exit the `nano` window.

Then submit the batch job again:
```bash
sbatch inference.batch
```

To view GPU usage statistics, type `cat error_llama_infer.%A` where `%A` is the job number. This will open the error file for viewing.


### Prompting with a Text File
This script runs inference as a Slurm batch job by reading a prompt from a text file and saving the model’s response to an output file. This approach is best suited for automation and large-scale experiments.

We will use `nano` to create the text prompt.
```bash
nano text_1.txt
```
Once the `nano` window opens, copy and paste the following text into the window.

```txt
"Describe how an AI system could assist in detecting early signs of harmful algal blooms. Identify the types of data required and potential sources of error. Explain how model outputs should be validated by domain experts.Provide a short analysis of risks and limitations. Present your answer in a structured list." 
```
Hit CTRL+O to save the file, enter, then CTRL+X to exit the `nano` window.

We will also use `nano` to create the batch file.
``` bash
nano text_inf.batch
```
Once the `nano` window opens, copy and paste the following text into the window.

```bash
#!/bin/bash
#SBATCH --job-name=llama_infer
#SBATCH --output=output_llama_infer.%A
#SBATCH --error=error_llama_infer_text.%A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:1


set -euo pipefail

module load apptainer

########################
# Paths
########################

# Path to your Container
CONTAINER="$HOME/llm/llama_cuda.sif"  

########################
# Run
########################

echo "Starting inference..."
time srun apptainer exec --nv -B /grps2:/grps2 "$CONTAINER" \
    llama-cli \
        -m /grps2/ualib/llm/phi-4-Q6_K.gguf \
        -n 512 \
        --threads "$SLURM_CPUS_PER_TASK" \
        --ctx-size 3000 \
        -ngl 999 \
        --single-turn \
        < $HOME/llm/text_1.txt > $HOME/llm/response_1.txt
```
Hit CTRL+O to save the file, enter, then CTRL+X to exit the `nano` window.

Next, use `sbatch` to run the text file prompt.
```bash
sbatch text_inf.batch
```
To check status of submitted job:
```bash
squeue -u {myBamaID}
```
To view output use the following and replace `%A` with the job number:
```bash
cat response_1.txt
```
Note that the time to run the model is listed at the bottom of the output file.
To view GPU usage statistics, type `cat error_llama_infer.%A` where `%A` is the job number. This will open the error file for viewing.

### Changing Model Parameters

In this section, we will improve the model’s performance by modifying the context size and the maximum number of generated tokens. The current model parameters are configured for shorter, more concise prompts, and they do not perform as well when handling more complex topics or longer requests.

By adjusting the model parameters, we can optimize factors such as response quality, creativity, speed, and overall scalability of the model.

The context size, `--ctx-size`,  determines how much your model can remember at once. This includes the user prompts, chat template text, and conversation history. If your context window is too small, it will cause the model to forget instructions, lose details, and contradict itself. 

The maximum number of tokens `-n`, determines how long the model's response is. If outputs are getting cut short, then we should increase `-n`. If the model rambles, then decrease `-n`.

In terminal, run : `nano text_inf.batch`

This will open a window that will allow you to adjust the batch file. We will make the following changes:

```bash
#!/bin/bash
#SBATCH --job-name=llama_infer
#SBATCH --output=output_llama_infer.%A
#SBATCH --error=error_llama_infer_text.%A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:1


set -euo pipefail

module load apptainer

########################
# Paths
########################

# Path to your Container
CONTAINER="$HOME/llm/llama_cuda.sif"

########################
# Run
########################

echo "Starting inference..."
time srun apptainer exec --nv -B /grps2:/grps2 "$CONTAINER" \
    llama-cli \
        -m $HOME/llm/models/phi-4-Q6_K.gguf \
        -n 512 \
        --threads "$SLURM_CPUS_PER_TASK" \
        --ctx-size 4096 \
        -ngl 999 \
        --single-turn \
        < $HOME/llm/text_1.txt > $HOME/llm/response_1.txt
```
Now, when we re-run the model, we will have a longer response with a summary statement. 
```bash
sbatch text_inf.batch
```
To view output:
```bash
cat response_1.txt
```

# Step 7: Running other models
We can also run OpenAI's gpt-oss 20b and 120b models using the same container. The gpt-oss models are autoregressive Mixture-of-Experts (MoE) transformers that can handle reasoning, tool use, and coding. In particular gpt-oss-120b can handle multi-step reasoning tasks. For more information, read the [model card](https://arxiv.org/pdf/2508.10925) and [technical report](https://cdn.openai.com/pdf/08b7dee4-8bc6-4955-a219-7793fb69090c/Technical_report__Research_Preview_of_gpt_oss_safeguard.pdf).
- **[OpenAI gpt-oss models](https://huggingface.co/collections/openai/gpt-oss)**  

We can now compare output from multiple models.

## Step 1. Request a GPU node
```bash
srun --pty --partition=gpu --qos=gpu --gpus=1 --cpus-per-task=4 --mem=48G --time=01:00:00 bash
```
## Step 2. Verify GPU status
```bash
nvidia-smi
```
## Step 3. Load module
```bash
module load apptainer
```
## Step 4. Create logs
```bash
mkdir -p llm_test_logs
```
## Step 5. Interact with models
Start a recording session and interact:
```bash
script -a llm_test_logs/phi4_session.log
apptainer exec --nv -B /grps2/ualib/llm:/llm $HOME/llm/llama_cuda.sif \
  llama-cli -m /llm/phi-4-Q6_K.gguf -ngl 999
```
During this session, ask these questions:
1. How is the weather today?

2. If all birds can fly, and a penguin is a bird, can a penguin fly?

3. Please provide a brief summary of the primary purpose and key outcomes of the Apollo 11 mission.

Enter `CTRL+C` to exit the llama-cli interactive session. Then type `exit` to stop recording.

**Now switch models.** Start a new recording session for the 20b model:
```bash
script -a llm_test_logs/20b_session.log
apptainer exec --nv -B /grps2/ualib/llm:/llm $HOME/llm/llama_cuda.sif \
  llama-cli -m /llm/gpt-oss-20b.gguf -ngl 999
```
Ask the same questions:
1. How is the weather today?

2. If all birds can fly, and a penguin is a bird, can a penguin fly?

3. Please provide a brief summary of the primary purpose and key outcomes of the Apollo 11 mission.

Enter `CTRL+C` to exit the interactive session; then type `exit` to stop recording. 

**Finally, switch to the 120b model.**
**Note about the 120b model:** The gpt-oss-120b model requires GPUs with at least 40GB VRAM (such as A100 or H100 GPUs). You may get an Out-Of-Memory error with the current resource request. Running 120b  is possible but we need to adjust `-ngl` down(`-ngl 10`) and it may be slow for interactive use.

```bash
script -a llm_test_logs/120b_session.log
apptainer exec --nv -B /grps2/ualib/llm:/llm $HOME/llm/llama_cuda.sif \
  llama-cli -m /llm/gpt-oss-120b.gguf -ngl 10
```
Ask the same questions:
1. How is the weather today?

2. If all birds can fly, and a penguin is a bird, can a penguin fly?

3. Please provide a brief summary of the primary purpose and key outcomes of the Apollo 11 mission.

Enter `CTRL+C` to exit the interactive session. Then type `exit` to stop recording.

## Step 6. Compare outputs
Verify the log files exist
```bash
ls -lh llm_test_logs/phi4_session.log
ls -lh llm_test_logs/20b_session.log
ls -lh llm_test_logs/120b_session.log
```
Then view the log files to compare output. 
```bash
cat llm_test_logs/phi4_session.log
cat llm_test_logs/20b_session.log
cat llm_test_logs/120b_session.log
```

## Troubleshooting & Tips

- For very large models (>50GB), you will need to download them to `$SCRATCH`
- Use `-t` to set the number of CPU threads, and `-ngl` to control GPU layers (higher = more GPU usage).
- `t/s` means **tokens per second**—the number of tokens processed or generated by the model each second. Higher values indicate faster inference speed.

## Considerations, Ethics, and Responsible Use of Language Models

Use only AI tools approved by [OIT](https://ai.ua.edu/ai-tools/approved-tools/) and follow OIT [guidelines and minimum AI security standards](https://oit.ua.edu/services/cybersecurity/minimum-security-standards/).

Sensitive or restricted data should not be used with generative  models; for more information and examples of sensitive data see [here](https://oit.ua.edu/services/cybersecurity/protect-ua/data-classification/). 

Students should check course policies before using AI. When conducting research, speak with your research advisor before using AI tools. Use language models responsibly. 

Consider first if your use-case is appropriate. Users should always verify outputs for accuracy and bias. Attribute any usage of AI or language models clearly, indicating that the output is machine-generated; and remember that language models can be helpful AI assistants, but they are computational tools, not humans.

### Additional Resources for Ethical and Responsible AI

- [Responsible AI at Microsoft](https://www.microsoft.com/en-us/ai/responsible-ai)
- [Microsoft Code of Conduct](https://huggingface.co/microsoft/phi-4-gguf/blob/main/CODE_OF_CONDUCT.md)
- [UA AI Teaching Network](https://uateachingacademy.ua.edu/ai-teaching-network/)  

## References
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama.cpp Discussions & Guides](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [HuggingFace Models](https://huggingface.co/models)
- [GGUF Format Info](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Apptainer](https://apptainer.org/docs/user/latest/introduction.html)