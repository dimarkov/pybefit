#!/bin/bash
for m in {1..10}
do
python run_fits_single.py --model $m --seed $m --device cpu
done
exit 0
