import torch
import numpy as np
from torch import Tensor
from typing import List, Union


class Mailbox(object):
    """A container for node mailbox messages."""

    def __init__(self, num_nodes: int, size: int, dim: int, device=None):
        self._size = size
        self._device = torch.device('cpu' if device is None else device)

        self._mail = torch.zeros((num_nodes, size, dim), device=self._device).squeeze(dim=1).pin_memory()
        # # default
        # self._mail = torch.zeros((num_nodes, size, dim), device=self._device).squeeze(dim=1)
        self._time = torch.zeros((num_nodes, size), device=self._device).squeeze(dim=1)
        if size > 1:
            self._next = torch.zeros(num_nodes, dtype=torch.long, device=self._device)

        self._mail_uniq_bids = torch.full((num_nodes, 1), -1) # mailbox 的uniq，对应的bids，初始化为-1
        self._mail_nbrs_bids = torch.full((num_nodes, 2), -1) # mailbox 的nbrs，对应的nids和bids，初始化为-1

    @property
    def mail(self) -> Tensor:
        return self._mail

    @property
    def time(self) -> Tensor:
        return self._time

    @property
    def device(self) -> torch.device:
        return self._device

    def dims(self) -> List[int]:
        return list(self.mail.shape[1:])

    def reset(self):
        self._mail.zero_()
        self._time.zero_()

    def get_nids_bids(self, unique_nodes):
        return self._mail_uniq_bids[unique_nodes], self._mail_nbrs_bids[unique_nodes] # 应该是两列的rounds

    def store(self, nids: Union[np.ndarray, Tensor], mail: Tensor, mail_ts: Tensor, uniq=None, nbrs=None, bid=None):
        if not isinstance(nids, Tensor):
            nids = torch.from_numpy(nids).long()
        nids = nids.to(self._device)
        mail = mail.detach().to(self._device)
        mail_ts = mail_ts.detach().to(self._device)
        if self._size == 1:
            self._mail[nids] = mail
            self._time[nids] = mail_ts
        else:
            pos = self._next[nids]
            self._mail[nids, pos] = mail
            self._time[nids, pos] = mail_ts
            self._next[nids] = torch.remainder(pos + 1, self._size)

        # if bid != None:
        #     # uniq
        #     self._mail_uniq_bids[uniq] = bid
        #     # nbrs
        #     # 取第1列
        #     import pdb;pdb.set_trace()
        #     self._mail_nbrs_bids[uniq, :1] = nbrs.unsqueeze(1)
        #     # 取第2列
        #     self._mail_nbrs_bids[uniq, 1:] = bid

    def move_to(self, device, **kwargs):
        if device is None or self._device == device:
            return
        self._mail = self._mail.to(device, **kwargs)
        self._time = self._time.to(device, **kwargs)
        self._device = device
