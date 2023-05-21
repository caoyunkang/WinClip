import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)  # 进程池

    # datasets = ['mvtec','visa']
    datasets = ['visa']
    for dataset in datasets:

        classes = dataset_classes[dataset]

        for cls in classes[:]:

            sh_method = f'python eval_WinCLIP.py ' \
                        f'--dataset {dataset} ' \
                        f'--class-name {cls} ' \

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()  # 等待进程结束

