python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model Transformer
python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model LSTM

python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model Transformer
python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model ViTag
python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model GRU
python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model RNN
# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model UNet


# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model Transformer
# python train_vispos_baseline.py train --model Baseline --gpus 0 --phase 2 --dataset vifi --dec_model Transformer
# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model Transformer


# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model LSTM
# python train_vispos_baseline.py train --model Baseline --gpus 0 --phase 2 --dataset vifi --dec_model LSTM
# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model LSTM

# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model Transformer
# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model ViTag
# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model GRU
# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model RNN
# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model UNet


# python train_vispos_baseline.py train --model Baseline --gpus 0 --phase 2 --dataset vifi --dec_model ViTag
# python train_vispos_baseline.py train --model Baseline --gpus 0 --phase 2 --dataset vifi --dec_model GRU
# python train_vispos_baseline.py train --model Baseline --gpus 0 --phase 2 --dataset vifi --dec_model RNN
# python train_vispos_baseline.py train --model Baseline --gpus 0 --phase 2 --dataset vifi --dec_model UNet


# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model ViTag
# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model GRU
# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model RNN
# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model UNet

# python train_vispos_shuffle.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model Transformer



# python train_vispos.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model ViTag
# python train_vispos_baseline.py train --model Baseline --gpus 0 --phase 2 --dataset vifi --dec_model ViTag
# python train_vispos_baseline.py train --model Vanilla --gpus 0 --phase 2 --dataset vifi --dec_model ViTag