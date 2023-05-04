# Dataset name: Inspec, SemEval2010, SemEval2017, DUC2001, nus, krapivin

dataset_name=SemEval2017 # ok


python3 main.py --dataset_dir ./data/$dataset_name --batch_size 128  \
--log_dir ./ --dataset_name $dataset_name
