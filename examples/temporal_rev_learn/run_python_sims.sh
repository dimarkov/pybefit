#!/bin/bash
for m in {1..19}
do
python run_sims.py -n 50 --model $m --seed $m --device gpu
done
exit 0