for seed in {1235..1239}
do
  for rpe in kerple_power_ap_ kerple_log
  do
    for dataset in arxiv github openwebtext2
    do
      python ./deepy.py train.py kerple_configs/local_setup.yml kerple_configs/train.yml kerple_configs/exp_configs/"$rpe"_"$dataset"_"$seed".yml
    done
  done
done
