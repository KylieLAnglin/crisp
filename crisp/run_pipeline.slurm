#!/bin/bash
#SBATCH --job-name=llama_pipeline_05_26_2025
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kylie.anglin@uconn.edu

set -e  # <-- Stop immediately if any command fails

# Load Python and activate environment
module load python/3.12.2
source /scratch/kla21002/kla21002/ollama_env_2025_05_22/bin/activate
export PYTHONPATH=/scratch/kla21002/kla21002/crisp:$PYTHONPATH
# cd /scratch/kla21002/kla21002/crisp # This is where you need to be

# Start Ollama instance (if needed)
apptainer instance start --nv /scratch/kla21002/kla21002/ollama ollama_instance

# Start the Ollama server in the background
apptainer exec instance://ollama_instance ollama serve &
# Wait for the server to start
sleep 5

# Run pipeline
echo "APE"
# python3 crisp/1_baseline_prompt/01_baseline_train.py
# python3 crisp/1_baseline_prompt/02_baseline_dev.py
# python3 crisp/1_baseline_prompt/03_fewshot_train.py
# python3 crisp/1_baseline_prompt/04_fewshot_dev.py
python3 /scratch/kla21002/kla21002/crisp/crisp/2_APE/03_APE_fewshot_train.py
python3 /scratch/kla21002/kla21002/crisp/crisp/2_APE/04_APE_fewshot_dev.py

echo "Pipeline is now complete."