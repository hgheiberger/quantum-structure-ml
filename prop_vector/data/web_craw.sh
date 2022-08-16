#!/bin/bash
#SBATCH --account=linhnk		# username to associate with job
#SBATCH --job-name=web_craw		# a desired name to appear alongside job ID in squeue
#SBATCH --gres=gpu:1 			# number of GPUs (per node)
#SBATCH --time=0-03:00			# time (DD-HH:MM)
#SBATCH --output="%x_%j.out"		# output file where all text printed to terminal will be stored
					# current format is set to "job-name_jobID.out"
nice -n 19 python urop.py 	# command or script to run; can use 'nvidia-smi' as a test
