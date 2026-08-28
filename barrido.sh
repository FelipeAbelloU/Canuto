#!/bin/bash
cd /bodega/FelipeAbello/Canuto
source venv/bin/activate
run () {   # $1=nombre  $2=flags entrenamiento  $3=flags eval
  echo "=== $1 INICIO $(date) ==="
  python scripts/train_gpu.py $2 --output "data/checkpoints/7b-$1"
  python scripts/evaluate.py --checkpoint "data/checkpoints/7b-$1" $3 --limit 150
  echo "=== $1 FIN $(date) ==="
}

run e2-lr2e4-r16  "--epochs 2 --lr 2e-4 --rank 16"  "--epochs 2 --lr 2e-4"
run e4-lr2e4-r16  "--epochs 4 --lr 2e-4 --rank 16"  "--epochs 4 --lr 2e-4"
run e3-lr1e4-r16  "--epochs 3 --lr 1e-4 --rank 16"  "--epochs 3 --lr 1e-4"
run e3-lr3e4-r16  "--epochs 3 --lr 3e-4 --rank 16"  "--epochs 3 --lr 3e-4"
run e3-lr2e4-r8   "--epochs 3 --lr 2e-4 --rank 8"   "--epochs 3 --lr 2e-4"
run e3-lr2e4-r32  "--epochs 3 --lr 2e-4 --rank 32"  "--epochs 3 --lr 2e-4"
