# Serial vs. Parallel Processing in Python Card Sorting Benchmark on HPC
Contributors: Rachel DuBose, Nate Pedigo, and Jacob Hoernlein
University of Alabama Libraries Research Computing Services

Last updated July 10, 2026

This Python program demonstrates the difference between **serial (single-core)** and **parallel (multi-core)** computation using a hypothetical exercise based on shuffling and processing card decks.

Learning Objectives:
- Logging into UAHPC
- Simulate a large number of independent tasks  
- Run them both **sequentially** and **in parallel**
- Measure performance differences   

This guide will also cover the instructions for submitting the script as a Slurm batch job on an HPC cluster, including environment setup and job monitoring.

In these exercises, we will be simulating a large number of decks of cards. Once we create each deck of cards, each deck will be shuffled. Then, each deck will be sorted by suit, then rank, with each card converted into a numeric value and all values summed.

Download the `cardsort` folder in the **Day 2 Exercise folder** and extract all files. It will contain: `cardsort_serial.py`, `cardsort_serial.batch`,`cardsort.py`, `cardsort.batch`, `mpi.py`, and `mpi.batch`. Once extracted, leave them in your Downloads folder.

---

## 1. Accessing UAHPC

1. First, make sure you are on the VPN.

To download the client visit [here](https://ua-app01.ua.edu/software/public/vpn/showFiles). Double-click the downloaded file. It is a silent install, but you may notice some flickering of icons. Wait approximately one minute and you should see a Cisco AnyConnect box pop up.

If you have already downloaded the client, open the Cisco AnyConnect VPN client. Click “Connect” to establish a connection. If a full URL is required, enter “uavpn.ua.edu/campus”.

Authenticate with myBamaID + password + Duo method in the “Second Password” field. In the “Second Password” field, you can enter `push` to receive a push notification to your Duo-registered device, `phone` to receive a phone call to your Duo registered phone, or you can enter a passcode retrieved from the Duo app. Users must have a Duo account to access the VPN.

You may need to click Yes to allow a certificate. After this step, you’re connected!

2. Open your terminal on Linux or Mac; on Windows, use PowerShell. Recall, one way to connect to a Linux server is to use SSH (Secure Shell). SSH is a network protocol that allows you to securely connect to a remote server over a network. You will need the hostname of the server, your username, and your password (or SSH key). For this session, the hostname is **uahpc.ua.edu**, the
username is your **myBamaID**, and password is your **myBama password**.You will need to connect to the UAHPC server using this command:

```bash
ssh {myBamaID}@uahpc.ua.edu
```
You will be prompted to complete Duo two-factor authentication and enter your myBama password. 
If it is your first time logging on, you may be asked to accept a fingerprint; type `yes`.

---

## 2. File transfer using SFTP

To transfer the files we need to and from the UAHPC, one option is to use SFTP (SSH File Transfer Protocol) which is a secure file transfer protocol that performs all operations over an encrypted ssh protocol. 

First, we need to make the correct directory.

```bash
mkdir cardsort
```
Once we have made the directory, we are going to navigate into the newly created directory, using:
```bash
cd cardsort
```

### Connecting to the SFTP Server

In a new terminal, you must first connect to the SFTP server using the following command: 
```bash
sftp {myBamaID}@uahpc.ua.edu
```
You will then be prompted to enter your password, and complete Duo two-factor authentication. After successfully connecting to the server you should see a system prompt like this:

```sftp
sftp>
```
Then, to transfer the files we will need for our exercise, we will execute the following command:
Remember to switch out {myBamaID} for your myBamaID.
```sftp
put -r C:\Users\myBamaID\Downloads\cardsort\*
exit
```
---
## 3. Serial batch script for HPC

Back in the original `ssh` terminal, use `ls` to verify the files are there.
```bash
ls
```
Now that we see our files, we will submit the serial version of the program.
We will be using the Slurm batch script (`cardsort_serial.batch`) with the following headers:

```bash
#!/bin/bash
#SBATCH --job-name=cardsort_serial        #Name of the job for identification
#SBATCH --ntasks=1                        #Number of tasks 
#SBATCH --cpus-per-task=1                 #Number of CPU cores allocated for the task 
#SBATCH --mem 4G                          #Amount of memory allocated for the job (4 GB in this case)
#SBATCH --time=02:00:00                   #Maximum time allowed for the job (2 hours in this case)
#SBATCH -p main                           #Partition to submit the job to (main in this case)
#SBATCH --qos main                        #Quality of Service for the job (main in this case)
#SBATCH -e errors_serial.%A               #File to save standard error output, with `%A` as the job ID
#SBATCH -o output_serial.%A               #File to save standard output, with `%A` as the job ID

module load miniconda3/base
conda create -n cards -y
conda activate cards
python ./cardsort_serial.py
```

To submit the job:
```bash
sbatch cardsort_serial.batch
```
To monitor the progress of your job use:
```bash
 squeue --me 
 ``` 
This will show you the status of your job in the queue. Once the job is complete, you can check the output files specified in the batch script for results and any potential errors.

## 4. Understanding the serial Python script
### a. Program Configuration

While this program is running, we will open and view `cardsort_serial.py` where we sort one deck of cards at a time.
```bash
cat cardsort_serial.py
```
Now that the script is open, let's examine each section. 
```python
NUM_DECKS = 5000000
SEED = 21
CHUNK_SIZE = 5000
```
Here is where we set a variety of parameters that will influence the execution of our program.
- **NUM_DECKS**: Total number of simulated decks of cards to shuffle (tasks)
- **SEED**: Base random seed for reproducibility
- **CHUNK_SIZE**: Number of tasks grouped together per batch

**Note:** in serial processing, the total work amount is to process 5 million decks (NUM_DECKS).
Each deck will be shuffled and processed regardless of chunk size. Chunk size only affects HOW the work is accomplished, not how much work is done. 

---

### b. Deck Representation

Here is where we establish the definition of our representative "deck of cards". 

```python
BASE_DECK = [(rank, suit) for suit in range(4) for rank in range(2, 15)]
```

- Each card is represented as a tuple `(rank, suit)`
- Ranks range from 2 to 14 (11–14 = Jack, Queen, King, Ace)
- Suits are encoded as integers 0–3
- Array indices in most coding languages **start at 0**, hence we have 0-3 representing a total of 4 suits.

---

### c. Shuffling the Deck

Using our random seed, shuffle the deck entirely and return the deck in its newly shuffled state. 

```python
def make_shuffled_deck(seed):
    rng = random.Random(seed)
    deck = BASE_DECK.copy()
    rng.shuffle(deck)
    return deck
```

- Uses a deterministic random generator (`seed`)
- Ensures reproducibility across runs
- Returns a shuffled deck

---

### d. Processing a Deck

Our actual "work" function that makes up our computational workload. 

```python
def work(deck):
    sorted_deck = sorted(deck, key=lambda c: (c[1], c[0]))
    return sum(r * 10 + s for r, s in sorted_deck)
```

Steps:
1. Sort cards by suit, then rank  
2. Convert each card into a numeric value  
3. Sum all values  

Summing the values of the sorted deck may seem odd, but it serves as a placeholder for a more complex processing task. Ultimately, the output is a single number representing the processed deck.

---

### e. Chunk Processing

Helper function to aggregate the work performed.

```python
def process_chunk(args):
    start, stop, seed = args
    out = []
    for i in range(start, stop):
        deck = make_shuffled_deck(seed + i)
        out.append(work(deck))
    return out
```

- Processes a **range of tasks**
- Each task gets a unique seed (`seed + i`)
- Returns a list of computed results
---

### f. Splitting Work into Chunks

Function for dividing up the work. Even with serial computing, there may be advantages to chunking including reducing memory usage and helping with model input constraints and any streaming compatibility.

```python
def chunk_ranges(n, chunk_size):
    for start in range(0, n, chunk_size):
        yield (start, min(start + chunk_size, n))
```
- Divides total tasks into manageable pieces

---
### g. Execute the program

```python
def main():
    chunks = [(start, stop, SEED) for start, stop in chunk_ranges(NUM_DECKS, CHUNK_SIZE)]

    # Serial execution
    t0 = time.perf_counter()
    serial_results = []
    for start, stop, seed in chunks:
        serial_results.extend(process_chunk((start, stop, seed)))
    t1 = time.perf_counter()

    serial_time = t1 - t0

    print(f"decks: {NUM_DECKS}")
    print(f"chunk_size: {CHUNK_SIZE}")
    print(f"serial_time: {serial_time:.4f} s")
    print(f"total_results: {len(serial_results)}")

if __name__ == "__main__":
    main()
```
Now that we have executed the serial script, let's look at the parallel workflow. 

## 5. Parallel batch script for HPC
There are a few changes for the parallel version.  We are also introducing multiple CPUs to help accomplish this task.
We will submit the parallel version of the program using the Slurm batch script (`cardsort.batch`):
```bash
#!/bin/bash
#SBATCH --job-name=card_sort           #Name of job 
#SBATCH --ntasks=1                     #Number of tasks
#SBATCH --cpus-per-task=32             #Number of CPU cores allocated for the task 
#SBATCH --mem 32G                      #Amount of memory allocated for the job
#SBATCH --time=00:30:00                #Maximum time allowed for the job 
#SBATCH -p main                        #Partition for the job
#SBATCH --qos main                     #Quality of Service for the job
#SBATCH -e errors.%A                   #File to save standard error output, with `%A` as the job ID
#SBATCH -o output.%A                   #File to save standard output, with `%A` as the job ID

module load miniconda3/base
conda activate cards


python ./cardsort.py
```

To submit the job:
```bash
sbatch cardsort.batch
```

To monitor the progress of your job use:
```bash
 squeue --me 
 ``` 
This will show you the status of your job in the queue.

---

## 6. Parallel Python script
### a. Program Configuration

While this program is running, we will open and view `cardsort.py` where we distribute the deck sorting across 32 CPUs.
```bash
cat cardsort.py
```
Now that the script is open, let's examine each section. 

```python
# ---- USER SETTINGS ----
NUM_DECKS = 5000000
SEED = 21
CHUNK_SIZE = 5000
DEFAULT_PROCESSES = 32
# -----------------------
```
Here, we set a variety of parameters including:
- **NUM_DECKS**: Total number of simulated decks of cards to shuffle (tasks).
- **DEFAULT_PROCESSES**: Number of worker processes.

### b. Deck Representation

This has not changed from the serial script. Here, we establish the definition of our representative "deck of cards".

```python
BASE_DECK = [(rank, suit) for suit in range(4) for rank in range(2, 15)]
```
- Each card is represented as a tuple `(rank, suit)`
- Ranks range from 2 to 14 (11–14 = Jack, Queen, King, Ace)
- Suits are encoded as integers 0–3
- Array indices in most coding languages **start at 0**, hence we have 0-3 representing a total of 4 suits.

---

### c. Shuffling the Deck

This section is exactly like the serial script. Using our random seed, shuffle the deck entirely and return the deck in its newly shuffled state.

```python
def make_shuffled_deck(seed):
    rng = random.Random(seed)
    deck = BASE_DECK.copy()
    rng.shuffle(deck)
    return deck
```
- Uses a deterministic random generator (`seed`)
- Ensures reproducibility across runs
- Returns a shuffled deck
---
### d. Processing a Deck

This section also stays the same. Below, is our "work" function that makes up our computational workload.

```python
def work(deck):
    sorted_deck = sorted(deck, key=lambda c: (c[1], c[0]))
    return sum(r * 10 + s for r, s in sorted_deck)
```
Steps:
1. Sort cards by suit, then rank  
2. Convert each card into a numeric value  
3. Sum all values  

Summing the values of the sorted deck may seem odd, but it serves as a placeholder for a more complex processing task. Ultimately, the output is a single number representing the processed deck.

---

### e. Chunk Processing

This section is identical to the Chunk processing in the serial script.

```python
def process_chunk(args):
    start, stop, seed = args
    out = []
    for i in range(start, stop):
        deck = make_shuffled_deck(seed + i)
        out.append(work(deck))
    return out
```

- Processes a **range of tasks**
- Each task gets a unique seed (`seed + i`)
- Returns a list of computed results
---
### f. Determines CPU usage
```python
def get_processes():
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    return DEFAULT_PROCESSES
```
- Detects available CPUs available for tasking
- Supports **Slurm cluster environments**
- Returns discovered number of CPUs or the default number of processes, 32 CPUs.

---
### g. Splitting Work into Chunks
```python
def chunk_ranges(n, chunk_size):
    for start in range(0, n, chunk_size):
        yield (start, min(start + chunk_size, n))
```
- Divides total tasks into manageable pieces

---
### h. Execute the program
```python
def main():
    processes = get_processes()
    chunks = [(start, stop, SEED) for start, stop in chunk_ranges(NUM_DECKS, CHUNK_SIZE)]

    # Parallel execution
    t0 = time.perf_counter()
    parallel_results = []
    with ProcessPoolExecutor(max_workers=processes) as ex:
        for chunk_result in ex.map(process_chunk, chunks):
            parallel_results.extend(chunk_result)
    t1 = time.perf_counter()

    parallel_time = t1 - t0
    throughput = NUM_DECKS / parallel_time

    print(f"decks: {NUM_DECKS}")
    print(f"cpus: {processes}")
    print(f"chunk_size: {CHUNK_SIZE}")
    print(f"parallel_time: {parallel_time:.4f} s")
    print(f"throughput: {throughput:.2f} decks/s")
    print(f"total_results: {len(parallel_results)}")

if __name__ == "__main__":
    main()
```
Examine output with:
```bash
cat output.%A
```
That ran quickly. Now we will examine what we can do to further experiment and refine the parallel script.

---
## Exercises

### 1. Increase number of decks
First, we were sorting 5 Million decks; let's increase the number of decks by an order of magnitude to 50 Million.

To do this:
```bash
nano cardsort.py
```
Then change NUM_DECKS to 50000000 and save the file using `CTRL+O`, `enter`, and `CTRL+X`. 

Now submit the job:
```bash
sbatch cardsort.batch
```

How much longer did that take?
### 2. Change number of CPUs
Now try changing the number of CPUs from 32 to 8 or 16.

To do this:
```bash
nano cardsort.py
```
Change `DEFAULT_PROCESSES = 32` to `DEFAULT_PROCESSES = 8` or `DEFAULT_PROCESSES = 16`.
Save the file using `CTRL+O`, `enter`, and `CTRL+X`.

You also need to change the batch script. 
First:
```bash
nano cardsort.batch
```
Change `--cpus-per-task=32` to `--cpus-per-task=8` or `--cpus-per-task=16`.
Save the file using `CTRL+O`, `enter`, and `CTRL+X`.

Now,submit the job:
```bash
sbatch cardsort.batch
```
How did that change how fast the program runs?

### 3. Change chunk size
Next, change the chunk size to 10000. To do this:
```bash
nano cardsort.py
```
Make CHUNK_SIZE = 10000 and save the file using `CTRL+O`, `enter`, and `CTRL+X`.

Now submit the job:
```bash
sbatch cardsort.batch
```

Did that improve the time?

### Bonus exercise- MPI
Now run the MPI style version. To do this:
```bash
sbatch mpi.batch
```
Examine the Python script with:
```bash
nano mpi.py
```
And view outputs where `%A` is the job number:
```bash
cat output_mpi.%A
```