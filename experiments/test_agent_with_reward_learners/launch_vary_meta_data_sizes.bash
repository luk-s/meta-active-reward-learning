#!/bin/bash
python launch_experiments.py --configs config_point_env_fpacoh_gp_PPO_meta_0010.py config_point_env_fpacoh_gp_PPO_meta_0020.py config_point_env_fpacoh_gp_PPO_meta_0050.py config_point_env_fpacoh_gp_PPO_meta_0100.py config_point_env_fpacoh_gp_PPO_meta_0500.py config_point_env_fpacoh_gp_PPO_meta_1000.py --params num_runs=30::int --globalparams num_cpus_per_job=8::int exec_time=23::int
