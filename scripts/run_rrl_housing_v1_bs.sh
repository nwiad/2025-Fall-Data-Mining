
WORLD_SIZE_1="0"

WORLD_SIZE_2="0@1"

WORLD_SIZE_8="0@1@2@3@4@5@6@7"

cd rrl/
# [Test Regression] MSE=13.1455, MAE=2.6763, R2=0.7978
# [Structure] #Edges = 410, Log(#Edges+eps) = 6.0162
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 2 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.001 \
#     -ki 0 \
#     -i ${WORLD_SIZE_2} \
#     -wd 0.0001 \
#     --task_type "regression"\
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# [Test Regression] MSE=13.1455, MAE=2.6763, R2=0.7978
# [Structure] #Edges = 410, Log(#Edges+eps) = 6.0162
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 4 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.001 \
#     -ki 0 \
#     -i ${WORLD_SIZE_2} \
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

# [Test Regression] MSE=20.1803, MAE=3.3205, R2=0.6897
# [Structure] #Edges = 157, Log(#Edges+eps) = 5.0562
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 16 \
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


# [Test Regression] MSE=31.8275, MAE=3.9686, R2=0.5106
# [Structure] #Edges = 98, Log(#Edges+eps) = 4.5850
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 32 \
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


# [Test Regression] MSE=475.7654, MAE=20.2601, R2=-6.3164
# [Structure] #Edges = 10, Log(#Edges+eps) = 2.3026
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 64 \
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

# [Test Regression] MSE=475.7654, MAE=20.2601, R2=-6.3164
# [Structure] #Edges = 10, Log(#Edges+eps) = 2.3026
# python3 experiment.py \
#     -d housing_train_v1 \
#     -bs 128 \
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