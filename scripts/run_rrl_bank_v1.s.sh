

WORLD_SIZE_8="0@1@2@3@4@5@6@7"

cd rrl/

# f1 = 73.1
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 32 \
#     -s 1@16 \
#     -e 50 \
#     -lrde 200 \
#     -lr 0.002 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best

# 1@16 -> 5@64
# f1 = 72.7
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 32 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.002 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best

# bs32 -> bs64, 1@16 -> 5@64
# f1 = 72.2
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 64 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.002 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best

# bs32 -> bs64, 1@16 -> 5@64, use_not
# f1 = 75.1
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 64 \
#     -s 5@64 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.002 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not

# bs32 -> bs64, 1@16 -> 10@64@32@16, use_not
# f1 = 46.8
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 64 \
#     -s 10@64@32@16 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.002 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best \
#     --use_not