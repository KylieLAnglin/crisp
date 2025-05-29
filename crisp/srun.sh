# _____________ log in _____________ 

# replace netid with your personal NetID
ssh -Y netid@hpc2.storrs.hpc.uconn.edu
# check that you were assigned a login node:
hostname

# _____________ set up files _____________ 

# I NOT CLONED then clone github repo
# don't clone the repo in a subfolder
# clone specific branch
git clone --single-branch --branch add-llama https://github.com/KylieLAnglin/crisp.git

# clone main
git clone https://github.com/KylieLAnglin/crisp.git

# I CLONED: 
# update the repo with any changes
# pull branch changes
git pull 
# pull main changes
git pull origin main

# in filezilla, upload the following files to the repo:
# copy library/secrets_template.py, rename to secrets.py, add GPT API key
# copy contents of OneDrive/Anglin, Kylie's files - crisp/data to 8_hpc/data
# copy the contents of OneDrive/Materials for LLM to 8_hpc/Materials for LLM

# _____________ request a job _____________

# hardware recs for llama 4 from: https://bizon-tech.com/blog/llama-4-system-gpu-requirements-running-locally?srsltid=AfmBOoow2UfYJfu-93BJbFUeOuIV0OcdENqt_HmxJQnMd6jtbYoEGQOZ
srun --partition=general-gpu --mem=64G --pty bash
hostname

# _____________ load software _____________

# load python version
module load python/3.12.2

# OPTIONAL STEP: build container - see APPENDIX

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

# OPTIONAL STEP: pull a model - see appendix

# check that the model speciied as PLATORM in start.py is downloaded to the container
apptainer exec instance://ollama_instance ollama list

# OPTIONAL: create virtual environment - see appendix

# activate the envrionment
source .venv/bin/activate

# install python packages to the virtual environment
pip3 install -r requirements.txt

# install crisp.library to the virtual environment
pip3 install -e .

# fix error with crisp.library
export PYTHONPATH=/home/$USER/crisp:$PYTHONPATH

# run the script
python3 crisp/1_baseline_prompt/01_baseline_train.py 
python3 crisp/1_baseline_prompt/run all.py

#python3 crisp/1_baseline_prompt/01_baseline_train.py 





# _____________ APPENDIX _____________


# _____________ build a container _____________

# save the container to scratch 
# navigate to scratch
# this might not be the exact path but it the general location o scratch
cd ../../scratch/PI_netID/PI_netID/crisp

# apptainer guide: https://apptainer.org/docs/user/main/cli.html
# build the container
apptainer build --force --docker-login --sandbox ollama/ docker://ollama/ollama:latest


# _____________ pull a model _____________

# download model
apptainer exec instance://ollama_instance ollama pull llama3.3
apptainer exec instance://ollama_instance ollama pull llama4
apptainer exec instance://ollama_instance ollama pull gemma3:12b

# _____________ create a virtual environment _____________

# set the location o the environment to the cloned repository
# it should be saved in the same location as requirements.txt
cd home/$USER/crisp/

# skip if the virtual environment is already created
# create a virtual environment for running the python scripts
python3 -m venv .venv


# _____________ sharing directory permissions _____________

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