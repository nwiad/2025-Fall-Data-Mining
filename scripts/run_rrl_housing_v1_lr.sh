
WORLD_SIZE_1="0"

WORLD_SIZE_2="0@1"

WORLD_SIZE_8="0@1@2@3@4@5@6@7"

cd rrl/


# [Test Regression] MSE=44.7694, MAE=4.6612, R2=0.3115
# [Structure] #Edges = 24, Log(#Edges+eps) = 3.1781
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.0001 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=20.5285, MAE=3.3569, R2=0.6843
# [Structure] #Edges = 148, Log(#Edges+eps) = 4.9972
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.0005 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=15.8078, MAE=2.9995, R2=0.7569
# [Structure] #Edges = 349, Log(#Edges+eps) = 5.8551
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.0008 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=15.4099, MAE=2.9062, R2=0.7630
# [Structure] #Edges = 409, Log(#Edges+eps) = 6.0137
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.001 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=12.1787, MAE=2.7031, R2=0.8127
# [Structure] #Edges = 532, Log(#Edges+eps) = 6.2766
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.005 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=12.4702, MAE=2.6957, R2=0.8082
# [Structure] #Edges = 478, Log(#Edges+eps) = 6.1696
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.008 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not


# [Test Regression] MSE=13.4993, MAE=2.7402, R2=0.7924.   !!!!
# [Structure] #Edges = 517, Log(#Edges+eps) = 6.2480
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.01 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=18.5975, MAE=3.6253, R2=0.7140
# [Structure] #Edges = 605, Log(#Edges+eps) = 6.4052
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.05 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=15.1130, MAE=2.9972, R2=0.7676
# [Structure] #Edges = 639, Log(#Edges+eps) = 6.4599
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 8 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.1 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not