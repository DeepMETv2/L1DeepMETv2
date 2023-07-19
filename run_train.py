import os

batch_size = 32 # default: 6
lr = 0.1 # default: 0.01
weight_decay = 0.001 # default: 0.001

process = 'ttbar'
ckpts = 'ckpts_{}_batch_{}_lr_{}_wd_{}'.format(process, batch_size, lr, weight_decay)

os.system('mkdir -p {}'.format(ckpts))

cmd = 'python train.py --data data_{} --ckpts {} --batch_size {} --lr {} --weight_decay {}'.format(process, ckpts, batch_size, lr, weight_decay)

print(cmd)
os.system(cmd)
