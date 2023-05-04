datasets=("Inspec" "SemEval2010" "SemEval2017" "DUC2001" "nus" "krapivin")

for dataset in ${datasets[*]}
    do
    python3 main.py --dataset_dir ./data/$dataset --batch_size 128 \
--log_dir ./ --dataset_name $dataset
    done

python3 summary.py