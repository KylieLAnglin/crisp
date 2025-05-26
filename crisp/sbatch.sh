# connect to vpn or campus wifi

# log into hpc

#!/bin/bash
#SBATCH --partition=lo-core                       # Name of partition
#SBATCH --ntasks=48                               # Request 48 CPU cores
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00                           # Job should run for up to 12 hours
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END                      # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --kylie.anglin@uconn.edu                  # Destination email address

# add commands
my app
