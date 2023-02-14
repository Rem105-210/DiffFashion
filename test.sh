#!/bin/sh
cd /media/zju/data1/style_trans/thism
echo "开始测试"

conda activate py39
for i in {1..27..1}
do 
    python main.py -i "../bag/${i}.jpg" --output_path "./op/output_reptile${i}" -tg "../fish/${i}.jpg" --use_range_restart --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 --use_colormatch --use_noise_aug_all
    wait
    echo "com ${i}"
done
echo "完成"
