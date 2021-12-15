#!/bin/bash
for m in {1..19}
do
python run_fits_dynpref.py --model $m --seed $m --device cpu
done
exit 0
