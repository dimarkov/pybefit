#!/bin/bash
for num in {1..10}
do
python run_sims.py -n 25 -g min -m 1 -nu $num
done
