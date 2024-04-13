import time
from collections import OrderedDict
import sys
from typing import Optional, Union, List
import typing

class TimeMarker():
    def __init__(self) -> None:
        self.time_marks: typing.OrderedDict[str, float] = OrderedDict()

    def mark(self, markname:str):
    # Be careful that if the mark already exists, it will not update and nothing happens.
        return self.time_marks.setdefault(markname, time.time())

    def get(self, markname:str)->Optional[float]:
        return self.time_marks.get(markname, None)
    
    def pop(self, markname:str):
        return self.time_marks.pop(markname)

    def reset(self):
        self.time_marks.clear()

    def get_all(self):
        return self.time_marks.copy()
    
    def get_interval(self, markname1:str, markname2:str)->float:
        return self.time_marks.get(markname2) - self.time_marks.get(markname1)

    def get_interval_name(self, markname1:str, markname2:str):
        return f"{markname1}->{markname2}"

    def get_all_intervals(self):
        intervals:typing.Dict[str, float] = dict()
        mark_list = list(self.time_marks.items())
        if len(mark_list) < 2:
            return intervals
        for i in range(len(mark_list)-1):
            interval_name = self.get_interval_name(mark_list[i][0], mark_list[i+1][0])
            interval = mark_list[i+1][1] - mark_list[i][1]
            intervals.setdefault(interval_name, interval)
        interval_name = self.get_interval_name(mark_list[0][0], mark_list[-1][0])
        interval = mark_list[-1][1] - mark_list[0][1]
        intervals.setdefault(interval_name, interval)
        return intervals
        
    def __str__(self) -> str:
        s = ""
        for k,v in self.time_marks.items():
            s = s + f"{k}:{v}\n"
        return s
    
def get_obj_size(obj:Union[bytes, List[bytes]]):
    empty_byte_size = sys.getsizeof(b'')
    if isinstance(obj, bytes):
        return sys.getsizeof(obj) - empty_byte_size
    else:
        sz = 0
        for b in obj:
            sz += get_obj_size(b)
        return sz