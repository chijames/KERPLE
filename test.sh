for seed in {1235..1239}
do
  for rpe in kerple_power_ap_ kerple_log
  do
    for dataset in arxiv github openwebtext2
    do
      for length in 512 1024 2048 4096 8192 16384
      do
        python ./deepy.py train.py kerple_configs/local_setup.yml kerple_configs/ex_eval.yml kerple_configs/exp_configs/"$rpe"_"$dataset"_"$seed".yml kerple_configs/lengths/length_"$length".yml
      done
    done
  done
done
