#b_12_32_96_96
python main.py --dir_data '../../../../../tmp/dataset-nctu' --model wdsr_b --pre_train wdsr_b_12_32_96_96_model_best.pt --scale 2 --n_resblocks 12 --n_feats 32 --block_feats 96 --res_scale 0.1 --patch_size 96 --data_range '1-800/1-100' --test_only
#b_16_64_32_96
python main.py --dir_data '../../../../../tmp/dataset-nctu' --model wdsr_b --pre_train wdsr_b_16_64_32_96_model_best.pt --scale 2 --n_resblocks 16 --n_feats 64 --block_feats 32 --res_scale 0.1 --patch_size 96 --data_range '1-800/1-100' --test_only
#b_16_32_128_96
python main.py --dir_data '../../../../../tmp/dataset-nctu' --model wdsr_b --pre_train wdsr_b_model_best.pt --scale 2 --n_resblocks 16 --n_feats 32 --block_feats 128 --res_scale 0.1 --patch_size 96 --data_range '1-800/1-100' --test_only
#baseline
python main.py --dir_data '../../../../../tmp/dataset-nctu' --pre_train download --scale 2 --data_range '1-800/1-100' --test_only