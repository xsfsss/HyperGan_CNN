import argparse
import os
import logging


from Train_cnn import train_gan
import torch


from hpbandster.core.nameserver import NameServer
from hpbandster.optimizers.hyperband import HyperBand
import hpbandster.core.result as hpres
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        null_handler = logging.NullHandler()
        self.logger.addHandler(null_handler)
        self.logger.propagate = False

    def compute(self, config, budget, *args, **kwargs):
        res = train_gan(config, opt)
        # 注意: train_gan 需要返回一个字典，包含一个 'loss' 键和一个 'info' 键
        print("check point res")
        #return res
        return {'loss': res, 'info': {}}

def get_configspace():
    cs = CS.ConfigurationSpace()

    # 例如, 对于学习率，我们可以使用对数均匀分布的采样
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-2, log=True)
    b1 = CSH.UniformFloatHyperparameter('b1', lower=0.0, upper=1.0)
    b2 = CSH.UniformFloatHyperparameter('b2', lower=0.0, upper=1.0)

    cs.add_hyperparameters([lr, b1, b2])
    return cs


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=192, help="size of the batches")
#parser.add_argument("--lr", type=float, default=0.00015, help="adam: learning rate")
#parser.add_argument("--b1", type=float, default=0.4, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--b2", type=float, default=0.65, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image samples")
parser.add_argument("--lr_decay", type=float, default=0.95, help="exponential learning rate decay")#学习率衰减策略
parser.add_argument('--model', type=str, default='cnn.pth', help='path to pretrained model')#加载预训练模型
opt = parser.parse_args()
print(opt)


def run_hyperband_optimization(worker_class, get_configspace_func, num_workers=1, n_iterations=20, min_budget=0.5, max_budget=0.9, run_id='optimal_run'):
    NS = NameServer(run_id=run_id, host='localhost', port=0)#这里localhost的意思是在本机运行，其实可以多机联动
    #print("check point NS created")
    ns_host, ns_port = NS.start()
    #print("check point NS start")
    workers = []
    for i in range(num_workers):
        #print("Check point 1")
        w = worker_class(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
        #print("Check point 2")
        w.run(background=True)
        workers.append(w)

    hb = HyperBand(configspace=get_configspace_func(),
                   run_id=run_id, nameserver=ns_host,
                   nameserver_port=ns_port,
                   min_budget=min_budget, max_budget=max_budget)
    res = hb.run(n_iterations=n_iterations)

    # 结束所有工作进程和nameserver
    for w in workers:
        w.shutdown()
    NS.shutdown()

    return res

def main():
    # 使用函数：
    res = run_hyperband_optimization(MyWorker, get_configspace)
    #print("check point res_main")
    all_runs = res.get_all_runs()
    best_run = res.get_incumbent_id()

    best_run_id = res.get_incumbent_id()#这里我修改了源文件，不然一直返回none我不知道怎么改
    print("run best id :")
    print(best_run_id)
    best_config = res.get_id2config_mapping()[best_run_id]['config']
    print(f"最佳配置: {best_config}")

if __name__ == "__main__":
    main()
