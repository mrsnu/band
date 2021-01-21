BASEPATH=/data/local/tmp

adb shell rm /data/local/tmp/log/*

SEC=1000
LOW=low
MID=mid
HIGH=high
for config in "low" # "low" "mid" "high"
do
  if [ $config == $LOW ]; then
    MODELS_PER_BATCH=16
  fi
  if [ $config == $MID ]; then
    MODELS_PER_BATCH=24
  fi
  if [ $config == $HIGH ]; then
    MODELS_PER_BATCH=36
  fi
  JSON=model_config.json.$config
  cp $JSON model_config.json
  adb push $JSON $BASEPATH
  for idx in 1 5 9 13 17 21 25 29
  do
    num_requests=$(($idx * 10))
    period=$(($MODELS_PER_BATCH*$SEC/($num_requests)))
    echo $period

    for planner in 2 # 0 2
    do
      adb shell /data/local/tmp/benchmark_model --period=$period --planner=$planner --json_path=$BASEPATH/$JSON --num_threads=4 --enable_op_profiling=true --write_profile_data=true --profile_data=$BASEPATH/model_profile.json
      mkdir -p experiments/$config
      path=experiments/$config/log_$planner\_$period
      adb pull /data/local/tmp/log $path
      adb shell rm /data/local/tmp/log/*
    #sleep 30
    done
  done 
done
