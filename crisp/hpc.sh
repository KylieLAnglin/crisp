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
# clean, fewshot_examples, prompts, temp

# request a job
# hardware recs for llama 4 from: https://bizon-tech.com/blog/llama-4-system-gpu-requirements-running-locally?srsltid=AfmBOoow2UfYJfu-93BJbFUeOuIV0OcdENqt_HmxJQnMd6jtbYoEGQOZ
srun --partition=general-gpu --mem=64G --pty bash
hostname

cd ../../scratch/PI_netID/PI_netID/crisp
# load python version
module load python/3.12.2

# apptainer guide: https://apptainer.org/docs/user/main/cli.html
# build the container
apptainer build --force --docker-login --sandbox ollama/ docker://ollama/ollama:latest
# start the container
# want to see ollama.toml at the same level as ollama/ container
# .toml is a config file to run the container faster
apptainer instance start --nv --bind ~/ollama.toml ollama/ ollama_instance
# start ollama in the container
apptainer exec instance://ollama_instance ollama serve

# _____________ terminal window 2 _____________
# login
ssh -Y netid@hpc2.storrs.hpc.uconn.edu
# go to running node
ssh node

# load python version
module load python/3.12.2

# download model
apptainer exec instance://ollama_instance ollama pull llama3.3
apptainer exec instance://ollama_instance ollama pull llama4
apptainer exec instance://ollama_instance ollama pull gemma3:12b

# BEFORE CREATING THE ENVRIONMENT
cd home/$USER/crisp/

# skip if the virtual environment is already created
# create a virtual environment for running the python scripts
python3 -m venv .venv

# activate the envrionment
source .venv/bin/activate

# install python packages to the virtual environment
pip3 install -r requirements.txt
# install own packages to the virtual environment
pip3 install -e .
# fix error with crisp package load
export PYTHONPATH=/home/$USER/crisp:$PYTHONPATH
# run the script
python3 crisp/1_baseline_prompt/00_baseline_prep.py 


# _____________ sharing directory _____________
# build container in scratch or shared - need more space for the larger model
# scratch and shared are only accessible via pi
# pi needs to give permission
cd ../.. 
# make a folder to save container
mkdir scratch/$USER/$USER/crisp/

# list info for files
# user = pi, needs to be done on pi's end
getfacl crisp/

cd scratch/$USER/$USER/crisp/

chmod go-rwx crisp/
setfacl -m "u:partner's_netid:rwx" crisp/
setfacl -dm "u:partner's_netid:rwx" crisp/
setfacl -dm "u:$USER:rwx" crisp/