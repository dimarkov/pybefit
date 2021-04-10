#!/bin/bash
for num in {2..10}
do
python run_sims.py -n 25 -nu $num
done