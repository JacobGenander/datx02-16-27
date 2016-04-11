#!/bin/bash

python -u translate.py --data_dir="/data" --article_file="articles_500000.txt" --title_file="titles_500000.txt" --train_dir="/seq3seq"
