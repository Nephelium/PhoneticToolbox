"""Source manager against the explicit loopback test instance, separate identity.

No real manager profile or installed EXE is changed. CA trust is per process.
"""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/'output/validation/web-launch/isolated/client'


def main():
    config=json.loads((PRIVATE/'client.json').read_text('utf-8'))
    assert config['manager_url']=='https://127.0.0.1:18768/api/v1/feedback'
    assert config['web_origin']=='http://127.0.0.1:18765'
    os.environ['SSL_CERT_FILE']=str(PRIVATE/'ca.pem')
    os.environ['LOCALAPPDATA']=str(PRIVATE/'profile')
    sys.path.insert(0,str(ROOT/'scripts'))
    from workbench_source import bind_sources
    bind_sources(ROOT)
    from ptb_desktop.feedback import FeedbackClient
    from ptb_desktop.platform_paths import user_data_root
    profile=user_data_root()/'feedback-manager'
    client=FeedbackClient(profile,admin=True,base_url=config['manager_url'])
    client.state={'token':config['manager_token'],'email':'','cursor':0};client.save()
    # Do not open an apparently connected manager when the tunnel is absent.
    client.request('webStatus',{});client.close()
    from ptb_desktop import feedback_manager
    original=feedback_manager.FeedbackManager
    class IsolatedManager(original):
        def setWindowTitle(self,title):
            super().setWindowTitle(title if title.startswith('隔离测试') else '隔离测试 · '+title)
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.setWindowTitle('隔离测试 · 网页审批与服务器监控')
    feedback_manager.FeedbackManager=IsolatedManager
    return feedback_manager.run(ROOT/'output/validation/web-launch/isolated/manager-web')


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        print('隔离测试管理端无法连接，请先恢复专用SSH隧道；正式反馈收件箱未改变。',file=sys.stderr)
        raise SystemExit(1) from None
