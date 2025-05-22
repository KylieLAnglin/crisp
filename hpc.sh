# _____________ terminal window 1 _____________
# replace netid with your personal NetID
ssh -Y netid@hpc2.storrs.hpc.uconn.edu
hostname

# don't clone the repo in a subfolder
git clone https://github.com/KylieLAnglin/crisp.git
# alt = clone specific branch
git clone --single-branch --branch add-llama https://github.com/KylieLAnglin/crisp.git

# if already cloned
# pull branch changes
git pull 
# pull main changes
git pull origin main

# set your working directory inside the repo
cd crisp
# check files loaded
ls

# upload files via filezilla
# copy the following folders from onedrive/crisp/data/ to github/hpc/data/
# clean, fewshot_examples, prompts

# request a job
# hardware recs for llama 4 from: https://bizon-tech.com/blog/llama-4-system-gpu-requirements-running-locally?srsltid=AfmBOoow2UfYJfu-93BJbFUeOuIV0OcdENqt_HmxJQnMd6jtbYoEGQOZ
srun --partition=general-gpu --mem=64G --pty bash
hostname

# load python version
module load python/3.12.5

# apptainer guide: https://apptainer.org/docs/user/main/cli.html
# build the container
apptainer build --force --docker-login --sandbox ollama/ docker://ollama/ollama:latest
# start the container
apptainer instance start ollama/ ollama_instance
# start ollama in the container
apptainer exec instance://ollama_instance ollama serve

# _____________ terminal window 2 _____________
# login
ssh -Y netid@hpc2.storrs.hpc.uconn.edu
# go to running node
ssh node

# download model
apptainer exec instance://ollama_instance ollama pull llama3.3
apptainer exec instance://ollama_instance ollama pull llama3.4

# create a virtual environment for running the python scripts
python3 -m venv llama4_env
# activate the envrionment
source llama4_env/bin/activate

# install python packages to the virtual environment
pip3 install -r requirements.txt
# install own packages to the virtual environment
pip3 install -e .
# run the script
python3 crisp/1_baseline_prompt/00_baseline_prep.py 