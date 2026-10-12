"""Owned real HTTP/SSE/local SMTP and hidden Qt feedback acceptance."""
import argparse
import asyncore
import base64
from datetime import datetime,timezone,timedelta
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage
from io import BytesIO
import json
from pathlib import Path
import re
import smtplib
import smtpd
import socket
import threading
import time

from workbench_source import bind_sources
from verification_artifacts import verification_run,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qt',action='store_true')
    parser.add_argument('--no-mail',action='store_true')
    options=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    bind_sources(root)
    from ptb_api.feedback import FeedbackService,create_feedback_app
    from ptb_desktop.feedback import FeedbackClient
    import uvicorn
    receipt={'success':False,'checks':[],'scope':'Windows current source; actual loopback HTTP/SSE and plaintext local SMTP fixture. No production mail, server or final EXE.'}
    with verification_run('feedback','integration','independent private SQLite; synthetic PNG; loopback HTTP and local SMTP; native DPAPI; optional hidden Qt') as (out,scratch):
        print(json.dumps({'out':str(out)}),flush=True)
        smtp_map={}
        mails=[]
        smtp_stop=threading.Event()
        class Mailbox(smtpd.SMTPServer):
            def process_message(self,peer,mailfrom,rcpttos,data,**kwargs):
                mails.append(BytesParser(policy=policy.default).parsebytes(data))
        receiver=Mailbox(('127.0.0.1',0),None,map=smtp_map)
        smtp_port=receiver.socket.getsockname()[1]
        def smtp_loop():
            while not smtp_stop.is_set():
                asyncore.loop(timeout=.05,count=1,map=smtp_map)
        smtp_thread=threading.Thread(target=smtp_loop,daemon=True)
        smtp_thread.start()
        class LocalSMTP:
            def send(self,recipient,subject,body,key,attachments=()):
                mail=EmailMessage()
                mail['From']='maintainer@example.org'
                mail['To']=recipient
                mail['Subject']=subject
                mail.set_content(body)
                for mime,data in attachments:
                    mail.add_attachment(data,maintype='image',subtype=mime.split('/')[1],filename='reply-image.jpg')
                with smtplib.SMTP('127.0.0.1',smtp_port,timeout=5) as client:
                    client.send_message(mail)
        now=[datetime(2026,10,10,8,tzinfo=timezone(timedelta(hours=8))).timestamp()]
        admin_key='integration-admin-test-only-123456789'
        service=FeedbackService(scratch/'feedback.sqlite',secret='integration-secret-test-only-123456789',admin_token=admin_key,mailer=None if options.no_mail else LocalSMTP(),clock=lambda:now[0],allow_unverified=options.no_mail)
        sock=socket.socket()
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(create_feedback_app(service),host='127.0.0.1',port=port,access_log=False,log_level='critical',timeout_graceful_shutdown=1))
        server_thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True)
        server_thread.start()
        base=f'http://127.0.0.1:{port}/api/v1/feedback'
        user=FeedbackClient(scratch/'user',base_url=base,test=True)
        admin=FeedbackClient(scratch/'admin',admin=True,base_url=base,test=True)
        events=[]
        windows=[]
        app=None
        def wait(condition,timeout=15):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if condition():return
                time.sleep(.05)
            raise AssertionError('Acceptance condition timed out')
        def mail_code(address):
            for message in reversed(mails):
                if message['To']==address:
                    text=message.get_body(preferencelist=('plain',)).get_content()
                    match=re.search(r'验证码：(\d{6})',text)
                    if match:return match.group(1)
            return None
        try:
            wait(lambda:server.started)
            if options.no_mail:
                user.request('openDevice',{'email':'user@example.org'})
                assert user.status()['connected'] and not user.status()['verified']
            else:
                challenge=user.request('verifyStart',{'email':'user@example.org'})
                wait(lambda:mail_code('user@example.org'))
                user.request('verifyFinish',{'challenge':challenge['challenge'],'code':mail_code('user@example.org')})
                assert user.status()['verified']
            image=BytesIO()
            from PIL import Image
            Image.new('RGB',(100,60),(63,108,168)).save(image,format='PNG')
            data=base64.b64encode(image.getvalue()).decode()
            first=user.request('send',{'text':'真实 HTTP 反馈与图片','images':[{'data':data}],'requestId':'integration-message-1'})
            admin.request('connect',{'baseUrl':base,'token':admin_key})
            summaries=admin.request('conversations',{})['conversations']
            owner=summaries[0]['id']
            admin.subscribe(events.append)
            now[0]+=3600
            wait(lambda:any(e.get('kind')=='daily_feedback' for e in events))
            assert sum(e.get('kind')=='daily_feedback' for e in events)==1
            service.daily_check()
            assert len([e for e in service.events(0) if e['kind']=='daily_feedback'])==1
            receipt['checks'].append('real SMTP verification received, native protected identity, HTTP message/image, server-owned daily check and real SSE notification')
            reply=admin.request('send',{'owner':owner,'text':'维护者已回复，邮件含图片。','images':[{'data':data}],'requestId':'integration-reply-1'})
            duplicate=admin.request('send',{'owner':owner,'text':'维护者已回复，邮件含图片。','images':[{'data':data}],'requestId':'integration-reply-1'})
            assert reply['id']==duplicate['id'] and duplicate['duplicate']
            if options.no_mail:
                assert mails==[] and not service.deliver_one()
            else:
                wait(lambda:any(m['Subject']=='PhoneticToolbox：您的反馈收到回复' for m in mails))
                delivered=next(m for m in mails if m['Subject']=='PhoneticToolbox：您的反馈收到回复')
                assert len(list(delivered.iter_attachments()))==1
            conversation=user.request('messages',{'after':0})
            assert conversation['messages'][-1]['text']=='维护者已回复，邮件含图片。'
            assert user.request('unread',{})['unread']==1
            user.request('read',{'through':reply['id']})
            assert user.request('unread',{})['unread']==0
            receipt['checks'].append('reply persisted once; received by application and real local SMTP with sanitized image attachment; unread cleared')
            if options.qt:
                from PyQt6.QtCore import QEventLoop,QTimer,Qt,QEvent,QCoreApplication
                from PyQt6.QtWidgets import QApplication
                from ptb_desktop.host import Workbench,register_scheme
                from ptb_desktop.feedback_manager import FeedbackManager
                from ptb_desktop.feedback_bridge import FeedbackBridge
                register_scheme()
                app=QApplication(['feedback-owned-QA'])
                def pause(ms=80):
                    loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
                def js(window,code):
                    loop=QEventLoop();values=[]
                    window.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()))
                    QTimer.singleShot(6000,loop.quit);loop.exec()
                    assert values,'Qt JS timeout'
                    return values[0]
                def until(window,code,seconds=25):
                    deadline=time.monotonic()+seconds
                    while time.monotonic()<deadline:
                        if js(window,code):return
                        pause()
                    raise AssertionError(code+'\n'+str(js(window,'document.body.innerText.slice(-1200)')))
                def click(window,text):
                    expression='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+')'
                    until(window,'!!('+expression+')')
                    js(window,expression+'.click()');pause()
                def fill(window,selector,value):
                    js(window,'(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
                w=Workbench(root/'frontend/dist',test=True,vocal_profile=scratch/'vocal')
                windows.append(w)
                w.feedback_bridge.close()
                w.channel.deregisterObject(w.feedback_bridge)
                qt_user=FeedbackClient(scratch/'qt-user',base_url=base,test=True)
                w.feedback_bridge=FeedbackBridge(scratch/'qt-user',w,client=qt_user)
                w.channel.registerObject('feedback',w.feedback_bridge)
                w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
                w.resize(1280,800);w.show()
                until(w,'!!document.querySelector(".app-shell")')
                js(w,'document.querySelector("button[title=关于]").click()');pause()
                assert js(w,'document.querySelector(".about-website")?.href')=='https://www.phonetictoolbox.com/'
                w.view.grab().save(str(out/'about-website.png'))
                from PyQt6.QtGui import QDesktopServices
                from PyQt6.QtCore import QObject,QUrl,QPoint,pyqtSlot
                from PyQt6.QtTest import QTest
                opened=[]
                class UrlProbe(QObject):
                    @pyqtSlot(QUrl)
                    def capture(self,url):opened.append(url.toString())
                probe=UrlProbe()
                QDesktopServices.setUrlHandler('https',probe,'capture')
                try:
                    point=js(w,'(()=>{const r=document.querySelector(".about-website").getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
                    QTest.mouseClick(w.view.focusProxy(),Qt.MouseButton.LeftButton,pos=QPoint(int(point[0]),int(point[1])))
                    until(w,'!!document.querySelector(".about-website")');pause(300)
                    assert opened==['https://www.phonetictoolbox.com/'],opened
                finally:QDesktopServices.unsetUrlHandler('https')
                click(w,'问题反馈')
                until(w,'!!document.querySelector(".feedback-verification")')
                fill(w,'input[type=email]','qt-user@example.org')
                if options.no_mail:
                    click(w,'进入反馈会话')
                else:
                    click(w,'发送验证码')
                    until(w,'!!document.querySelector("input[autocomplete=one-time-code]")')
                    deadline=time.monotonic()+15
                    while not mail_code('qt-user@example.org') and time.monotonic()<deadline:pause()
                    assert mail_code('qt-user@example.org')
                    fill(w,'input[autocomplete=one-time-code]',mail_code('qt-user@example.org'))
                    click(w,'验证并进入会话')
                until(w,'!!document.querySelector(".thread-compose textarea")')
                until(w,'document.querySelector(".thread-profile img")?.naturalWidth>0')
                assert js(w,'document.querySelector(".thread-profile strong").textContent')=='井韶子'
                fill(w,'.thread-compose textarea','Qt 用户的真实反馈')
                click(w,'发送')
                until(w,'document.querySelector(".thread-timeline")?.textContent.includes("Qt 用户的真实反馈")')
                w.view.grab().save(str(out/'user-light.png'))
                qt_admin=FeedbackClient(scratch/'qt-admin',admin=True,base_url=base,test=True)
                qt_admin.request('connect',{'baseUrl':base,'token':admin_key})
                manager=FeedbackManager(root/'frontend/dist',root=scratch/'qt-manager',client=qt_admin,tray=False)
                windows.append(manager)
                manager.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
                manager.resize(1280,800);manager.show()
                until(manager,'[...document.querySelectorAll(".conversation-item")].some(e=>e.textContent.includes("qt-user@example.org"))')
                js(manager,'[...document.querySelectorAll(".conversation-item")].find(e=>e.textContent.includes("qt-user@example.org")).click()')
                until(manager,'document.querySelector(".thread-timeline")?.textContent.includes("Qt 用户的真实反馈")')
                fill(manager,'.thread-compose textarea','Qt 管理员回复已送达')
                click(manager,'发送回复')
                until(manager,'document.querySelector(".thread-timeline")?.textContent.includes("Qt 管理员回复已送达")')
                click(w,'刷新')
                until(w,'document.querySelector(".thread-timeline")?.textContent.includes("Qt 管理员回复已送达")')
                until(manager,'document.querySelector(".message-delivery")?.textContent.includes('+json.dumps('仅发送到应用' if options.no_mail else '已交给邮件服务器')+')')
                until(manager,'document.querySelector(".message-avatar.author-avatar")?.naturalWidth>0')
                assert '井韶子' in js(manager,'document.querySelector(".thread-timeline").textContent')
                js(w,'document.querySelector(".mail-toggle input").click()')
                until(w,'!document.querySelector(".mail-toggle input").checked&&!document.querySelector(".mail-toggle input").disabled')
                assert not service.messages(qt_user.state['owner'])['mailNotifications']
                fill(manager,'.thread-compose textarea','关闭邮件后继续交流')
                click(manager,'发送回复')
                until(manager,'document.querySelector(".thread-timeline")?.textContent.includes("关闭邮件后继续交流")')
                click(w,'刷新')
                until(w,'document.querySelector(".thread-timeline")?.textContent.includes("关闭邮件后继续交流")')
                assert service.messages(qt_user.state['owner'])['messages'][-1]['mailState']=='disabled'
                assert not service.deliver_one()
                w.view.grab().save(str(out/'user-mail-off.png'))
                manager.view.grab().save(str(out/'manager-light.png'))
                js(manager,'document.querySelector("button[aria-label=切换深色主题]").click()');pause(150)
                manager.view.grab().save(str(out/'manager-dark.png'))
                click(manager,'标记已解决')
                until(manager,'document.querySelector(".thread-toolbar")?.textContent.includes("已解决")')
                js(manager,'[...document.querySelectorAll(".conversation-item")].find(e=>e.querySelector("strong")?.textContent==="user@example.org").click()')
                until(manager,'!!document.querySelector("button[aria-label=放大反馈图片]")')
                js(manager,'document.querySelector("button[aria-label=放大反馈图片]").click()')
                until(manager,'!!document.querySelector("dialog[open] img")')
                js(manager,'document.querySelector("dialog[open] button[aria-label=关闭对话框]").click()')
                receipt['checks'].append('hidden actual Qt: About entry, email form/code, user send, separate manager receive/reply, author name/avatar, persistent user mail switch with suppressed reply delivery, user sync, resolve, light/dark screenshots; no tray notification shown during test')
            if options.no_mail:
                receipt['checks']=[c.replace('real SMTP verification received','SMTP disabled; native device identity').replace('real local SMTP with sanitized image attachment','SMTP disabled, no messages sent').replace('email form/code','contact email without verification') for c in receipt['checks']]
            receipt.update(success=True,mailMode='disabled' if options.no_mail else 'local-fixture',localMailCount=len(mails),serverEventCount=len(service.events(0)))
        finally:
            for window in reversed(windows):
                if hasattr(window,'quitting'):window.quitting=True
                if hasattr(window,'closing'):window.closing=True
                window.close()
                window.page.deleteLater()
            if app:
                QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
                app.processEvents()
                for window in reversed(windows):window.deleteLater()
                QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
                app.processEvents()
            user.close();admin.close()
            if admin.event_thread:admin.event_thread.join(3)
            server.should_exit=True
            server_thread.join(5)
            sock.close()
            smtp_stop.set();receiver.close();smtp_thread.join(2)
            write_json(out/'report.json',receipt)
            print(json.dumps({'success':receipt['success'],'out':str(out),'checks':receipt['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
