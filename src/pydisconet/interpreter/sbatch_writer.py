#### Redundant code. We can throw it off.
def shell_script_cpu(save_path, commands,job_name):
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --cluster=htc
#SBATCH --array=1-{str(len(commands))}
#SBATCH -t 0-06:00:00
#SBATCH --output={save_path}/slurm_outs/{job_name}/%x_%A_%a.out
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
##SBATCH --mail-user=swk25@pitt.edu
##SBATCH --mail-type=END,FAIL

# module purge
# module load gcc/8.2.0
# module load python/anaconda3.10-2022.10
# source activate /ix/djishnu/Swapnil/.conda/envs/coauth_env/

"""
    return sbatch_script

def shell_script_gpu(save_path, commands,job_name):
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={1 if '3_' in job_name else 4}
#SBATCH --mem-per-cpu=40G ### Will allocate 40*4 to each gpu ## To preven cuda oom
#SBATCH --partition=l40s,a100,gtx1080,a100_nvlink
#SBATCH --array=1-{str(len(commands))}
#SBATCH -t 0-06:00:00
#SBATCH --output={save_path}/slurm_outs/{job_name}/%x_%A_%a.out
##SBATCH --mail-user=swk25@pitt.edu
##SBATCH --mail-type=END,FAIL

# module purge
# module load gcc/8.2.0
# module load python/anaconda3.10-2022.10
# source activate /ix/djishnu/Swapnil/.conda/envs/coauth_env/

"""
    return sbatch_script