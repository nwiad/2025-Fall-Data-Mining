
WORLD_SIZE_1="0"

WORLD_SIZE_2="0@1"

WORLD_SIZE_8="0@1@2@3@4@5@6@7"

cd rrl/

# f1 = 73.1
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 32 \
#     -s 1@16 \
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

# lr0.002 -> lr0.004
# f1 = 72.6
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 32 \
#     -s 1@16 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.004 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best

# bs32 -> bs64
# f1 = 71.5
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 64 \
#     -s 1@16 \
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

# bs32 -> bs64, lr0.002 -> lr0.004
# f1 = 71.8
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 64 \
#     -s 1@16 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.004 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best

# bs32 -> bs128, lr0.002 -> lr0.004
# f1 = 71.0
# python3 experiment.py \
#     -d bank_train_v1 \
#     -bs 128 \
#     -s 1@16 \
#     -e 100 \
#     -lrde 200 \
#     -lr 0.004 \
#     -ki 0 \
#     -i ${WORLD_SIZE_8} \
#     -wd 0.0001 \
#     --print_rule \
#     --master_address "127.0.0.1" \
#     --master_port 12345 \
#     --save_best