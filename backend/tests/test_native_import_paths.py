"""Exercise a real CPython extension past Win32 MAX_PATH without changing resources."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.skipif(sys.platform != 'win32', reason='Win32 native extension loading')
def test_deep_native_extension_keeps_python_resource_paths(tmp_path):
    folder=tmp_path
    while len(str(folder))<285:
        folder=folder/'native-library-directory'
    folder.mkdir(parents=True)
    source=Path(importlib.util.find_spec('_decimal').origin)
    shutil.copyfile(source,folder/source.name)
    (folder/'resource_probe.py').write_text('from pathlib import Path\nRESOURCE=Path(__file__).with_name("resource.txt").read_text()\n','utf8')
    (folder/'resource.txt').write_text('same resource','utf8')
    backend=Path(__file__).resolve().parents[1]/'src'
    code='''import sys,json
sys.path.insert(0,sys.argv[1])
from ptb_worker.source_runtime import prepare_native_import_paths
prepare_native_import_paths();prepare_native_import_paths()
sys.path.insert(0,sys.argv[2])
import _decimal,resource_probe
print(json.dumps(dict(result=str(_decimal.Decimal('1.2')+_decimal.Decimal('3.4')),
                     extension=_decimal.__file__,module=resource_probe.__file__,resource=resource_probe.RESOURCE)))
'''
    result=subprocess.run([sys.executable,'-I','-B','-X','utf8','-c',code,str(backend),str(folder)],capture_output=True,text=True,encoding='utf8',timeout=30)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['result']=='4.6' and value['resource']=='same resource'
    assert Path(value['extension']).samefile(folder/source.name)
    assert value['module']==str(folder/'resource_probe.py')
    assert not value['extension'].startswith('\\\\?\\')
