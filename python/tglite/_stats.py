import time
from pathlib import Path


class TimeTable(object):
    def __init__(self):
        self.csv = None
        self.reset_epoch()
        self.reset_batch()
        self.reset_model()

    def reset_epoch(self):
        """Set all time records to zeros"""
        self.t_epoch = 0.0
        self.t_loop = 0.0
        self.t_eval = 0.0
        self.t_forward = 0.0
        self.t_backward = 0.0
        self.t_sample = 0.0
        self.t_prep_batch = 0.0
        self.t_prep_input = 0.0
        self.t_post_update = 0.0
        self.t_mem_update = 0.0
        self.t_time_zero = 0.0
        self.t_time_nbrs = 0.0
        self.t_self_attn = 0.0
        self.t_aggregate_total = 0.0
        self.t_dedup_total = 0.0
        self.t_load = 0
        self.t_load_dup = 0
        self.t_load_zero = 0

    def reset_batch(self):
        """Set some time records to zeros"""
        self.t_forward_list = []
        self.t_backward_list = []
        self.t_prep_batch_list = []

    def reset_model(self):
        self.t_block = 0
        self.t_next = []
        self.t_dedup = []
        self.t_cache = []
        self.t_sample_list = []
        self.t_preload = 0
        self.t_aggregate = 0

    def start(self):
        # Uncomment for better breakdown timings
        #torch.cuda.synchronize()
        return time.perf_counter()

    def elapsed(self, start):
        # Uncomment for better breakdown timings
        #torch.cuda.synchronize()
        return time.perf_counter() - start

    def print_epoch(self, prefix='  '):
        """Print the timing breakdown of different components in an epoch"""
        lines = f'' \
            f'{prefix}epoch | total:{self.t_epoch:.6f}s loop:{self.t_loop:.6f}s eval:{self.t_eval:.6f}s\n' \
            f'{prefix} loop | forward:{self.t_forward:.6f}s backward:{self.t_backward:.6f}s sample:{self.t_sample:.6f}s prep_batch:{self.t_prep_batch:.6f}s prep_input:{self.t_prep_input:.6f}s post_update:{self.t_post_update:.6f}s\n' \
            f'{prefix} comp | mem_update:{self.t_mem_update:.6f}s time_zero:{self.t_time_zero:.6f}s time_nbrs:{self.t_time_nbrs:.6f}s self_attn:{self.t_self_attn:.6f}s\n' \
            f'{prefix}  wyq |aggregate:{self.t_aggregate_total:.6f}s {prefix} dedup:{self.t_dedup_total:.6f}s\n' \
            f'{prefix}  wyq |preload_dup:{0 if self.t_load == 0 else (self.t_load_dup / self.t_load) * 100}% preload_zero:{0 if self.t_load == 0 else (self.t_load_zero / self.t_load) * 100}%\n '
        print(lines, end='')

    def print_batch(self, prefix='      '):
        prep_batch_str = []
        for num in self.t_prep_batch_list:
            prep_batch_str.append(f"{num:.6f}s")
        forward_str = []
        for num in self.t_forward_list:
            forward_str.append(f"{num:.6f}s")
        backward_str = []
        for num in self.t_backward_list:
            backward_str.append(f"{num:.6f}s")
        # lines = f'' \
        #     f'{prefix}batch_prep_batch:{prep_batch_str}\n' \
        #     f'{prefix}batch_forward:{forward_str}\n' \
        #     f'{prefix}batch_backward:{backward_str}\n'
        lines = f'' \
            f'{prefix}batch_forward:{forward_str}\n' \
            f'{prefix}batch_backward:{backward_str}\n'
        print(lines, end='')

    def print_model(self, prefix='          '):
        next_str = []
        for num in self.t_next:
            next_str.append(f"{num:.6f}s")
        dedup_str = []
        for num in self.t_dedup:
            dedup_str.append(f"{num:.6f}s")
        cache_str = []
        for num in self.t_cache:
            cache_str.append(f"{num:.6f}s")
        sample_str = []
        for num in self.t_sample_list:
            sample_str.append(f"{num:.6f}s")
        lines = f'' \
            f'{prefix}model_block:{self.t_block:.6f}s\n' \
            f'{prefix}model_next:{next_str}\n' \
            f'{prefix}model_dedup:{dedup_str}\n' \
            f'{prefix}model_cache:{cache_str}\n' \
            f'{prefix}model_sample:{sample_str}\n' \
            f'{prefix}model_preload:{self.t_preload:.6f}s\n' \
            f'{prefix}model_aggregate:{self.t_aggregate:.6f}s\n'
        print(lines, end='')

    def csv_open(self, path):
        """Close the opened file (if any) and open a new file in write mode"""
        self.csv_close()
        self.csv = Path(path).open('w')

    def csv_close(self):
        """Close the opened file (if any)"""
        if self.csv is not None:
            self.csv.close()
            self.csv = None

    def csv_write_header(self):
        """Write the header line to the CSV file"""
        header = 'epoch,total,loop,eval,' \
            'forward,backward,sample,prep_batch,prep_input,post_update,' \
            'mem_update,time_zero,time_nbrs,self_attn'
        self.csv.write(header + '\n')

    def csv_write_line(self, epoch):
        """Write a line of timing information to the CSV file"""
        line = f'{epoch},{self.t_epoch},{self.t_loop},{self.t_eval},' \
            f'{self.t_forward},{self.t_backward},{self.t_sample},{self.t_prep_batch},{self.t_prep_input},{self.t_post_update},' \
            f'{self.t_mem_update},{self.t_time_zero},{self.t_time_nbrs},{self.t_self_attn}'
        self.csv.write(line + '\n')


# Global for accumulating timings.
tt = TimeTable()
