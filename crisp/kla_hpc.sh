# Step 1: Open cisco
# Step 2: Open a terminal, ssh into the HPC
ssh -Y kla21002@hpc2.storrs.hpc.uconn.edu
hostname

# Step 3: Git pull
cd crisp
git pull

# Step 4: Request a job
# big job
srun --partition=general-gpu --mem=64G --pty bash
# COPY hostname (e.g., gpu14)
# small job
# srun -t 00:30:00 --mem=8G --pty bash

# load python version
module load python/3.12.2

# Step 5: start the container
apptainer instance start /scratch/kla21002/kla21002/crisp/ollama ollama_instance



# _____________ terminal window 2 _____________
# login
ssh -Y kla21002@hpc2.storrs.hpc.uconn.edu
# go to running node
ssh node
