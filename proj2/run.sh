predicted_csv="$1"

python3 bonus.py data/train.csv model/bonus_bpr_r4.h5 bpr -r 4 -Ts "$predicted_csv"