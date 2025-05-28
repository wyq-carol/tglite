import torch
import os
import tglite.config
import subprocess
import math

def get_node_percent(counts):
    return 0.1

def parse_and_output_nodes(file_path, node_percent, log_dir, log_name):
    sorted_pairs = torch.load(file_path)
    
    # 拆分 nodes 和 counts
    nodes = [node for _, node in sorted_pairs]
    counts = [count for count, _ in sorted_pairs]  
    
    # 检查 nodes 和 counts 的长度是否一致
    assert len(nodes) == len(counts), "nodes 和 counts 的长度不一致"
    
    node_percent = get_node_percent(counts)
    
    # 输出前 node_percent 的 nodeid
    new_file_path = os.path.join(log_dir, f"hotID_nodes_{log_name}.pt")
    torch.save(nodes[0:math.floor(len(nodes) * node_percent)], new_file_path)


if __name__ == "__main__":
    # 配置参数
    node_percent = 0.1  # 假设我们想要前 10% 的节点
    log_dir = f"/home/volume/tglake_res"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    for i in range(len(tglite.config.DATA)):
        DATA = tglite.config.DATA[i]
        BATCH_SIZE = tglite.config.BATCH_SIZE[i]
        N_LAYERS = tglite.config.N_LAYERS[i]
        N_NBRS = tglite.config.N_NBRS[i]
        N_HEADS = tglite.config.N_HEADS[i]
        log_name = f"DATA_{DATA}_BS_{BATCH_SIZE}_NLAYER_{N_LAYERS}_NBR_{N_NBRS}_NHEAD_{N_HEADS}"
    

        file_path = os.path.join(log_dir, f"nodes_{log_name}.pt")
    
        # 加载保存的张量
        if not os.path.exists(file_path):
            os.chdir("/home/tglite/examples/")
            os.environ['PYTHONPATH'] = '/home/tglite:' + os.environ.get('PYTHONPATH', '')
            # 运行训练脚本
            train_command = (
                f"python tgn/train.py -d {DATA} --seed 0 --prefix exp "
                f"--epochs 1 --bsize {BATCH_SIZE} --n-threads 8 "
                f"--n-layers {N_LAYERS} --n-heads {N_HEADS} --n-nbrs {N_NBRS} "
                "--sampling recent "
                "--on-heter 0 --all-on-gpu 0 --on-statistic 1"
            )
            subprocess.run(train_command, shell=True, check=True)
            # 再次检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件 {file_path} 仍然不存在，请检查训练脚本是否正确生成了文件")
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        # 调用函数
        parse_and_output_nodes(file_path, node_percent, log_dir, log_name)