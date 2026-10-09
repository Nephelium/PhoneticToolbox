"""Read either historical one-file or persistent-cache distribution bytes."""
import json
from pathlib import Path
import sys
import tarfile
from PyInstaller.archive.readers import CArchiveReader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'desktop/src'))
from ptb_desktop.startup_cache import Payload


class DistributionReader:
    def __init__(self, executable):
        self.archive=CArchiveReader(str(executable));self.cache=None;self.values={}
        self.toc=self.archive.toc
        if 'cache-payload.json' in self.toc:
            self.cache=json.loads(self.archive.extract('cache-payload.json'))
            self.payload=Payload(executable,self.cache)
            # Per-file compressed sizes are not meaningful for a solid XZ
            # stream. Attribute compressed bytes to the three streams only.
            self.toc={name.removeprefix('_internal/'):(0,0,row['size'],0,'x') for name,row in self.cache['applicationFiles'].items()}
            for name,blob in [('runtime-payload.tar.xz','science'),('host-payload.tar.xz','host'),('application-payload.tar.xz','apps')]:
                self.toc[name]=(0,self.cache['blobs'][blob]['size'],0,0,'x')

    def extract(self,name):
        if self.cache is None:return self.archive.extract(name)
        blob={'runtime-payload.tar.xz':'science','host-payload.tar.xz':'host','application-payload.tar.xz':'apps'}.get(name)
        if blob:
            with self.payload.open(blob) as file:return file.read()
        if name not in self.values:
            wanted={'desktop-bundle.json','host-archive.json','host-files.json','runtime-files.json','source-snapshot.json',name}
            with self.payload.open('apps') as source,tarfile.open(fileobj=source,mode='r|xz') as archive:
                for item in archive:
                    key=item.name.removeprefix('_internal/')
                    if key in wanted and item.isfile():self.values[key]=archive.extractfile(item).read()
        if name not in self.values:raise KeyError(name)
        return self.values[name]
