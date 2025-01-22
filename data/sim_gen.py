from utils.data import _generate_few_shot_data_files

for data in ['wmoz', 'dstc2', 'sim_joint']:
    for data_ratio in [10, 20, 50]:
        _generate_few_shot_data_files(data_dir="downstream_sim/mnt/efs/data/da/" + data, data_ratio=data_ratio, num_runs=10, task='da')
