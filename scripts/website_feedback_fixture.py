"""Loopback-only browser test fixture; real feedback service, in-memory mail sink."""
from pathlib import Path
import argparse,json,re,secrets,socket,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend/src'))
from fastapi.staticfiles import StaticFiles
from ptb_api.feedback import FeedbackService,create_feedback_app
import uvicorn

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scratch',type=Path,required=True);args=parser.parse_args()
    scratch=args.scratch.resolve();assert scratch.is_relative_to(ROOT/'output/validation/website')
    class Mailbox:
        def __init__(self):self.sent=[]
        def send(self,*args):self.sent.append(args)
    mail=Mailbox();service=FeedbackService(scratch/'feedback.sqlite',secret=secrets.token_urlsafe(32),admin_token=secrets.token_urlsafe(32),mailer=mail)
    app=create_feedback_app(service,background=False)
    @app.get('/__fixture/code')
    def code():
        while service.deliver_one():pass
        return {'code':re.search(r'验证码：(\d{6})',mail.sent[-1][2]).group(1)}
    @app.post('/__fixture/reply')
    def reply():
        with service.db() as db:owner=db.execute('SELECT id FROM identities').fetchone()[0]
        service.send(owner,'admin','收到啦，网页和应用都能继续这段会话。<img src=x onerror=alert(1)>',[],'fixture-admin-reply')
        return {'ok':True}
    @app.get('/__fixture/state')
    def state():
        with service.db() as db:
            users=db.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0]
            attachments=db.execute('SELECT COUNT(*) FROM attachments').fetchone()[0]
            owner=db.execute('SELECT id FROM identities').fetchone()
        result=service.messages(owner[0]) if owner else {}
        return {'userMessages':users,'attachments':attachments,'mailNotifications':result.get('mailNotifications')}
    @app.get('/site-stats.json')
    def stats():return {'schema':1,'startedAt':'2026-10-12T00:00:00Z','pageViews':1,'downloads':{'total':0,'portable':0,'installer':0}}
    app.mount('/',StaticFiles(directory=ROOT/'temporary-site',html=True))
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
    print(json.dumps({'port':sock.getsockname()[1]}),flush=True)
    server=uvicorn.Server(uvicorn.Config(app,access_log=False,log_level='error'))
    server.run(sockets=[sock])
if __name__=='__main__':main()
