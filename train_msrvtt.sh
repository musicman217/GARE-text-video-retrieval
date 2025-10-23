export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:/home/username/gare

DATA_PATH=/home/username/gare/data/MST-VTT
python -m torch.distributed.launch \
--master_port 29510 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/3fps_videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--temp 3 \
--alpha 2 \
--beta 0.07 \
--lambda_dir 0.01 \
--lambda_epsilon 0.01 \
--lambda_lower 0.5
