#!/bin/bash

for i in {0..11}
do
    python test_ForeCA.py --obj_id $i
done
