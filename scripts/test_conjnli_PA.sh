python run_PA.py --model_type roberta --model_name_or_path ./output/ConjNLI_PA --srl_model_path ./output/srl_bert --do_eval --do_lower_case --task_name ConjNLI --data_dir data/NLI/ --max_seq_length 128 --per_gpu_eval_batch_size=32   --per_gpu_train_batch_size=32   --learning_rate 2e-5 --num_train_epochs 3 --output_dir output/ConjNLI_PA