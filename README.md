# TrajPred
# ViFiDataSetPreProcessing

To run, we use ViFi dataset and JRDB dataset for 
we also preprocess a version for OOSTraj, the download link is [Google Drive] 
JRDB/jrdb.pkl
vifi_dataset_gps/vifi_data.pkl

python train.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model Transformer  --learning_rate 0.001

The training log is under ./checkpoints/ 