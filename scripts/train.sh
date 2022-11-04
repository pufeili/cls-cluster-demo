#! /bin/bash
cd "$(dirname "$PWD")" || exit
echo "$PWD"
# shellcheck disable=SC2046
echo -e  '\n' $(date +%F%n%T) '\n'
python train.py --alpha1 0.0 --alpha2 0.0 --alpha3 10.0 --alpha4 0.0  --epoch 50
python train.py --alpha1 0.0 --alpha2 0.0 --alpha3 0.0  --alpha4 10.0  --epoch 50
python train.py --alpha1 1.0 --alpha2 0.0 --alpha3 10.0 --alpha4 0.0 --epoch 50
python train.py --alpha1 0.0 --alpha2 1.0 --alpha3 0.0  --alpha4 10.0 --epoch 50
python train.py --alpha1 1.0 --alpha2 1.0 --alpha3 0.0 --alpha4 10.0 --epoch 50
python train.py --alpha1 1.0 --alpha2 1.0 --alpha3 10.0 --alpha4 0.0 --epoch 50
python train.py --alpha1 1.0 --alpha2 1.0 --alpha3 10.0 --alpha4 10.0 --epoch 50
# shellcheck disable=SC2046
echo -e  '\n' $(date +%F%n%T) '\n'