Lệnh train model: python main.py --arch inid --dataset inid --data_path "Path to data" --save_path ./save/pre_trained --test_batch_size 256

Lệnh chạy evaluate: python main.py --arch inid --dataset inid --save_path .\save\evaluate\ --data_path D:\UIT\NCKH\Dataset\IoT_Network_Intrusion_Dataset\ --test_batch_size 256 --evaluate

Lệnh bfa random flip bit: python main.py --arch inid --dataset inid --data_path D:\UIT\NCKH\data\ --test_batch_size 128 --n_iter 100 --k_top 100 --save_path ./save/attack_random --print_freq 50 --resume D:\UIT\NCKH\BFA\save\model_best.pth.tar --workers 8 --ngpu 1 --gpu_id 1 --fine_tune --reset_weight --bfa --random_bfa

Lệnh bfa pbs: python main.py --arch inid --dataset inid --data_path D:\UIT\NCKH\data\ --test_batch_size 128 --n_iter 100 --k_top 100 --save_path ./save/attack--print_freq 50 --resume D:\UIT\NCKH\BFA\save\model_best.pth.tar --workers 8 --ngpu 1 --gpu_id 1 --fine_tune --reset_weight --bfa
