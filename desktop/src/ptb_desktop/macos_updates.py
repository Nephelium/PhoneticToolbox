"""Keep the macOS development bundle outside Windows update channels."""
from .updates import UpdateService,UpdateError
from .updates_bridge import UpdatesBridge

class MacUpdateService(UpdateService):
    def start_maintenance(self):return None
    def check(self,**options):
        raise UpdateError('PLATFORM_UNAVAILABLE','当前 Mac 测试版请通过安装包更新。')
    def download(self,*args,**options):
        raise UpdateError('PLATFORM_UNAVAILABLE','当前 Mac 测试版请通过安装包更新。')
    def apply(self,*args,**options):
        raise UpdateError('PLATFORM_UNAVAILABLE','当前 Mac 测试版请通过安装包更新。')

class MacUpdatesBridge(UpdatesBridge):
    def __init__(self,parent=None,**options):
        super().__init__(parent,service=MacUpdateService(**options))
