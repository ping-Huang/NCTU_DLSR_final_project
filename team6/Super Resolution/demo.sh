python main.py --data_test Demo --scale 2 --pre_train download --test_only --save_results --data_range '1-800/1-10'--cpu

python main.py --data_dir ./ --scale 2 --model wdsr_b --pre_train wdsr_b_model_best.pt --test_only --save_results --data_range '1-800/1-10'