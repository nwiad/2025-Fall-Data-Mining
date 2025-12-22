
WORLD_SIZE_1="0"

WORLD_SIZE_2="0@1"

WORLD_SIZE_8="0@1@2@3@4@5@6@7"

cd rrl/

# 75.1
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

# 75.4
python3 experiment.py \
    -d bank_train_v1 \
    -bs 64 \
    -s 5@512 \
    -e 100 \
    -lrde 20 \
    -lr 0.005 \
    -ki 0 \
    -i ${WORLD_SIZE_8} \
    -wd 0.0001 \
    --print_rule \
    --master_address "127.0.0.1" \
    --master_port 12345 \
    --save_best \
    --use_not