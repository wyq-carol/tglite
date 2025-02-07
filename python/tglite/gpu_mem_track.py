import gc
import datetime
import inspect

import torch
import numpy as np

r, ma, mr, mr_s, mr_l = 0, 0, 0, 0, 0
count = 0
count_memory_stats = 0

def pack_hook(x):
    global gpumemtracker
    gpumemtracker.track()
    print("Packing", x)
    tmp = x
    gpumemtracker.track()
    return tmp

def unpack_hook(x):
    print("Unpacking", x)
    return x

def sep(num):
    if num % 2 ** 20 == 0:
        return f"{num} = {num // 2 ** 20}MB"
    else:
        return f"{num} ≈ {num / 2 ** 20:.4f}MB"

def memory_stats(inputfile, inputline, device=0):
    return
    d = torch.cuda.memory_stats(device)

    print(f'cur small_pool {sep(d["allocated_bytes.small_pool.current"]).rjust(20)}')
    print(f'cur large_pool {sep(d["allocated_bytes.large_pool.current"]).rjust(20)}')

    global r, ma, mr, mr_s, mr_l
    global count, count_memory_stats
    last_r, last_ma, last_mr, last_mr_s, last_mr_l = r, ma, mr, mr_s, mr_l
    r = d["requested_bytes.all.current"]
    ma = d["allocated_bytes.all.current"]
    mr = d["reserved_bytes.all.current"]
    mr_s, mr_l = d["segment.small_pool.current"], d["segment.large_pool.current"]

    mat = d["active_bytes.all.current"]
    miat = d["inactive_split_bytes.all.current"]

    if mr_s - last_mr_s == 1 and mr_l - last_mr_l == 0:
        cur_mr_tag = 'new segment belong to small pool'
    elif mr_s - last_mr_s == 0 and mr_l - last_mr_l == 1:
        cur_mr_tag = 'new segment belong to large pool'
    elif mr_s - last_mr_s == 0 and mr_l - last_mr_l == 0:
        cur_mr_tag = 'no new segment'
    else:
        # 1. self.mem_cell(mail, mem) _VF                             ===update mem===
        # 2. (only all on GPU) torch.cos(self.w(is_zero_tensor, ts))  ===time encode===*2?
        # 3. nfeat_map(nfeat)                                         ===before aggr===
        # 4. (only all on GPU) self.out_fc(h_out)                     ===final output===
        if count < 7:
            count += 1
            cur_mr_tag = f'new segments belongs to {mr_s - last_mr_s} small pools, {mr_l - last_mr_l} large pools'
        else:
            raise ValueError
    mr_tag = f'small_pool({mr_s})   large_pool({mr_l})'
    mat_tag = f'small_pool({d["active.small_pool.current"]})   large_pool({d["active.large_pool.current"]})'
    miat_tag = f'small_pool({d["inactive_split.small_pool.current"]})   large_pool({d["inactive_split.large_pool.current"]})'

    if mat + miat != mr:
        print(f"***mat: {mat}, miat: {miat}, mr: {mr}***")
    assert mat + miat == mr  # 已分配显存 + 未分配显存 = Segments 总和
    assert mat == ma
    print("")
    print(f"count_memory_stats {count_memory_stats}: {inputfile} at line {inputline}")
    count_memory_stats += 1
    print(f"operation requested memory  : {sep(r-last_r).rjust(20)}")
    print(f"operation allocated memory  : {sep(ma-last_ma).rjust(20)}")
    print(f"operation reserved  memory  : {sep(mr-last_mr).rjust(20)}    {cur_mr_tag}")
    print(f"total     reserved  memory  : {sep(mr).rjust(20)}    {mr_tag}")
    print(f"total     active    memory  : {sep(mat).rjust(20)}    {mat_tag}")
    print(f"total     inactive  memory  : {sep(miat).rjust(20)}    {miat_tag}")

dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16/8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = 8/8 # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571

def get_mem_space(x):
    try:
        ret = dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
    return ret

class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, detail=True, path='', verbose=False, device=0):
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'
        self.verbose = verbose
        self.begin = True
        self.device = device

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def get_tensor_usage(self):
        sizes = [np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) for tensor in self.get_tensors()]
        return np.sum(sizes) / 1024**2

    def get_allocate_usage(self):
        return torch.cuda.memory_allocated() / 1024**2

    def clear_cache(self):
        gc.collect()
        # torch.cuda.empty_cache()

    def print_all_gpu_tensor(self, file=None):
        for x in self.get_tensors():
            print(x.size(), x.dtype, np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2, file=file)

    def track(self):
        """
        Track the GPU memory usage
        """
        frameinfo = inspect.stack()[1]
        where_str = frameinfo.filename + ' line ' + str(frameinfo.lineno) + ': ' + frameinfo.function

        with open(self.gpu_profile_fn, 'a+') as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                        f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")
                self.begin = False

            if self.print_detail is True:
                ts_list = [(tensor.size(), tensor.dtype) for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x),
                                    tuple(x.size()),
                                    ts_list.count((x.size(), x.dtype)),
                                    np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2,
                                    x.dtype) for x in self.get_tensors()}
                for t, s, n, m, data_type in new_tensor_sizes - self.last_tensor_sizes:
                    f.write(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')
                for t, s, n, m, data_type in self.last_tensor_sizes - new_tensor_sizes:
                    f.write(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')

                self.last_tensor_sizes = new_tensor_sizes

            f.write(f"\nAt {where_str:<50}"
                    f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                    f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")

global gpumemtracker
gpumemtracker = MemTracker() # todo