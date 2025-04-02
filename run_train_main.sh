#python3 scripts/classifier__single_seg/main.py --name 'cifar-10' --mode 'train' --model_name 'VGG16' --dataset.model_name 'VGG16' --created_model.model_name 'VGG16'

#python3 scripts/main.py --name 'cifar-10' --mode 'train' --model_name 'ResNet18' --dataset.model_name 'ResNet18' --created_model.model_name 'ResNet18'

#python3 scripts/main.py --name 'cifar-10' --mode 'train' --model_name 'ResNet34' --dataset.model_name 'ResNet34' --created_model.model_name 'ResNet34'

#python3 scripts/main.py --name 'cifar-10' --mode 'train' --model_name 'ResNet50' --dataset.model_name 'ResNet50' --created_model.model_name 'ResNet50'

#python3 scripts/main.py --name 'cifar-10' --mode 'train' --model_name 'ResNet101' --dataset.model_name 'ResNet101' --created_model.model_name 'ResNet101'

#python3 scripts/main.py --name 'cifar-10' --mode 'train' --model_name 'ResNet152' --dataset.model_name 'ResNet152' --created_model.model_name 'ResNet152'

#python3 scripts/main_seg.py --mode train --dataset.mode train --created_model.lr 1e-4 --train.num_epoch 200 --train.batch_size 64 --train.num_freq_save 50

#python3 scripts/main_seg.py --mode test --dataset.mode test  --created_model.lr 1e-4 --train.batch_size 32

#python3 scripts/main_seg.py --name_data 'isbi' --dataset.name_data 'isbi' --mode train --dataset.mode train --created_model.lr 1e-4 --train.num_epoch 200 --train.batch_size 1 --train.num_freq_save 50

#python3 scripts/main_seg.py --name_data 'isbi' --dataset.name_data 'isbi' --mode test --dataset.mode test  --created_model.lr 1e-4 --train.batch_size 1

#python3 scripts/main_seg_ver2.py --mode train --dataset.mode train --created_model.lr 1e-4 --train.num_epoch 300 --train.batch_size 64 --train.num_freq_save 50

#python3 scripts/main_seg_ver2.py --mode test --dataset.mode test --created_model.lr 1e-4 --train.batch_size 32

#python3 scripts/main_cyclegan.py --mode train --dataset.mode train --train.num_epoch 300

#python3 scripts/main_cyclegan.py --mode test --dataset.mode test --train.batch_size 1

python3 scripts/classifier__single_seg/main.py --name 'mnist' --mode 'train' --dataset.mode 'train' --model_name 'VGG16' --dataset.model_name 'VGG16' --created_model.model_name 'VGG16'

python3 scripts/classifier__single_seg/main.py --name 'mnist' --mode 'test' --dataset.mode 'test' --model_name 'VGG16' --dataset.model_name 'VGG16' --created_model.model_name 'VGG16'

echo "✅ 모든 학습이 완료되었습니다!"

# 'isbi' , 'pascal' , 'preprocessed'
