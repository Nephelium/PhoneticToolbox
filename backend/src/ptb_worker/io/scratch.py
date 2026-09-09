"""Host-owned scratch capability, exact reservations before file creation.

This is local-operation accounting. M01-F must bind it to durable owner/fencing
and P07 quota before exposing scientific jobs; no existing operations are enabled.
"""
from pathlib import Path
from uuid import uuid4
import os
import stat
from .limits import LimitError


def no_links(path):
    for item in [path,*path.parents]:
        if item.exists() or item.is_symlink():
            st=item.lstat()
            if stat.S_ISLNK(st.st_mode) or getattr(st,'st_file_attributes',0)&0x400:
                raise ValueError('Reparse paths are not allowed')


class Scratch:
    def __init__(self, parent, budget):
        if type(budget)!=int or budget<0:raise ValueError('Invalid scratch budget')
        parent=Path(parent).absolute();no_links(parent)
        if not parent.is_dir(): raise ValueError('Host scratch parent must exist')
        self.root=parent/('m01-'+uuid4().hex);self.root.mkdir()
        self.budget=budget;self.used=0;self.files={}

    def create(self,data,suffix):
        if suffix not in ['.wav','.json','.xlsx','.sqlite']: raise ValueError('Unsupported scratch kind')
        if self.used+len(data)>self.budget: raise LimitError('scratch_budget_exceeded')
        no_links(self.root)
        self.used+=len(data)
        path=self.root/(uuid4().hex+suffix);self.files[path]=len(data)
        try:
            with path.open('xb') as f:
                for i in range(0,len(data),65536):f.write(data[i:i+65536])
                f.flush();os.fsync(f.fileno())
        except BaseException:
            self.remove(path);raise
        return path

    def remove(self,path):
        if path not in self.files or path.parent!=self.root: raise ValueError('Not an owned scratch resource')
        no_links(self.root)
        path.unlink(missing_ok=True)
        self.used-=self.files.pop(path)

    def close(self):
        for path in list(self.files):self.remove(path)
        self.root.rmdir()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
