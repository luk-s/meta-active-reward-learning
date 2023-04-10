# Meta active reward learning
A repository containing the code for the Research in Data Science project, done during my Master studies at ETH Zürich. See the `project_report.pdf` file for more information about the project

## Installation

#### Creating virtual environment
This project was developed using Python 3.9. It is recommended to install this repository in a virtual environment.
Make sure you have [Python 3.9](https://www.python.org/downloads/release/python-390/) installed on your machine. Then, initialize your virtual environment in this folder, for example via the command 
```bash
python3.9 -m venv .venv
```
You can activate the virtual environment via the command
```bash
source .venv/bin/activate
```
#### Installing the required packages
All required packages can be installed via the command
```bash
pip install -e .
```
#### Euler setup
On the [Euler cluster](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) you need to first load the correct Python modules:
```bash
env2lmod
module load gcc/8.2.0
module load python/3.9.9
```
and then you can follow the same steps as stated above. After having set up the virtual environment on Euler, you can load all the required modules and activate the virtual environment automatically by running the command
```bash
source euler_setup.bash
```
## Usage
#### Setting the Python path
In order for the experiments to run properly, you might need to set the PYTHONPATH variable. This can be done with the following command:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```
## Reproducing the experiments
The experiment scripts were designed to be run on the Euler cluster. Make sure that you have access to the cluster and that you set everything up as described in the [Euler setup](<#euler-setup> "Euler setup") section. The experiments can be reproduced by running the following script:

#### Reproducing the main experiments
Enter the following folder:
```bash
cd experiments/test_agent_with_reward_learners
```
and then run the following commands. For the 2D environment:
```bash
python launch_experiments.py --configs 2d/config_point_env_goal_simple_gp_SAC.py 2d/config_point_env_goal_fpacoh_gp_SAC.py --globalparams num_cpus_per_job=8::int exec_time=23::int  
```
For the 3D environment:
```bash
python launch_experiments.py --configs 3d/config_point_env_goal_3d_simple_gp_SAC_new.py 3d/config_point_env_goal_3d_fpacoh_gp_SAC_new.py --globalparams num_cpus_per_job=8::int exec_time=23::int
```

#### Reproducing the noise separation experiments
Enter the following folder:
```bash
cd experiments/test_agent_with_reward_learners
```
and then run the following commands. For the 2D environment:
```bash
python launch_experiments.py --configs 2d/config_point_env_goal_simple_gp_SAC_separate.py 2d/config_point_env_goal_fpacoh_gp_SAC_separate.py --globalparams num_cpus_per_job=8::int exec_time=23::int separate_randomness=True::bool
```
For the 3D environment:
```bash
python launch_experiments.py --configs 3d/config_point_env_goal_3d_simple_gp_SAC_new_separate.py 3d/config_point_env_goal_3d_fpacoh_gp_SAC_new_separate.py --globalparams num_cpus_per_job=8::int exec_time=23::int separate_randomness=True::bool 
```

#### Finding the best hyperparameters for the reward learner
Feel free to change the hyperparameter ranges inside the specified config files before running the hyperparameter search. Enter the following folder:
```bash
cd experiments/test_reward_learners
```
and then run the following commands. For the 2D environment you can run the following commands:
```bash
python hyperparam_tuning.py --configs 2d/config_HYP_point_env_goal_simple_gp.py

python hyperparam_tuning.py --configs 2d/config_point_env_goal_fpacoh_gp.py
```
For the 3D environment:
```bash
python hyperparam_tuning.py --configs 3d/config_HYP_point_env_goal_3d_simple_gp.py

python hyperparam_tuning.py --configs 3d/config_HYP_point_env_goal_3d_fpacoh_gp.py
```

#### Finding the best hyperparameters for the reinforcement learning agent
Feel free to change the hyperparameter ranges inside the specified config files before running the hyperparameter search. Enter the following folder:
```bash
cd experiments/test_agent_with_reward_learners
```
and then run the following commands. For the 2D environment you can run the following commands:
```bash
python hyperparam_tuning.py --configs 2d/config_HYP_point_env_goal_simple_gp_SAC.py
```
For the 3D environment:
```bash
python hyperparam_tuning.py --configs 3d/config_HYP_point_env_goal_3d_simple_gp_SAC.py

```

## Reproducing the plots
In order to create the plots of your experiment, enter the directory of your experiment and then run the following command:
```bash
python plot_results.py
```
The program will display a list of experiment ids which are currently stored on your machine. Example:
```
Found the following runs:
  [0]: ca5f0f6526cf40a2ae88e530c81c9d3c (started at 2022-10-03 19:03:38)
  [1]: e2eef6708a8e47ed81301bdbfdbce6f9 (started at 2022-10-03 19:08:25)
  [2]: c21bd7d3069d41709b9eb441786d73ba (started at 2022-10-03 19:09:50)
  [3]: 215f7ad1646143b5951f32188d8a0369 (started at 2022-10-03 22:58:24)
  [4]: 8b6893b127ca41adb4ea3ba5f5772581 (started at 2022-10-03 23:02:04)
  [5]: 69ec337f27ef42dd877ec163bfd77b25 (started at 2022-10-26 09:34:11)
Please select the id of the run you want to use (0 - 6):
```
Type in the id of the run for which you would like to create the plots.
