
WORLD_SIZE_1="0"

WORLD_SIZE_2="0@1"

WORLD_SIZE_8="0@1@2@3@4@5@6@7"

cd rrl/

python3 experiment.py \
    -d housing_train_v1 \
    -bs 64 \
    -s 5@64 \
    -e 100 \
    -lrde 200 \
    -lr 0.002 \
    -ki 0 \
    -i ${WORLD_SIZE_8} \
    -wd 0.0001 \
    --print_rule \
    --master_address "127.0.0.1" \
    --master_port 12345 \
    --save_best \
    --use_not