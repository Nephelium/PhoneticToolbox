"""Private SMTP adapter; never import a user's batch-mail script."""
import ast
from email.message import EmailMessage
from pathlib import Path
import smtplib
import ssl


def settings_from_script(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    names = {'SMTP_SERVER', 'SMTP_PORT', 'SENDER_EMAIL', 'SENDER_PASSWORD'}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                values[name] = ast.literal_eval(node.value)
    if set(values) != names or type(values['SMTP_PORT']) is not int:
        raise ValueError('SMTP settings unavailable in the selected source')
    if any(not isinstance(values[k], str) or not values[k] or '\r' in values[k] or '\n' in values[k]
           for k in names - {'SMTP_PORT'}):
        raise ValueError('Invalid SMTP settings')
    return dict(host=values['SMTP_SERVER'], port=values['SMTP_PORT'], user=values['SENDER_EMAIL'],
                sender=values['SENDER_EMAIL'], password=values['SENDER_PASSWORD'], ssl=True)


class SMTPMailer:
    def __init__(self, settings):
        self.settings = settings

    def connection(self):
        s = self.settings
        if s.get('ssl', True):
            client = smtplib.SMTP_SSL(s['host'], int(s.get('port', 465)), timeout=15,
                                     context=ssl.create_default_context())
        else:
            client = smtplib.SMTP(s['host'], int(s.get('port', 587)), timeout=15)
            client.starttls(context=ssl.create_default_context())
        try:
            client.login(s['user'], s['password'])
        except Exception:
            client.close()
            raise
        return client

    def check(self):
        with self.connection() as client:
            client.noop()

    def send(self, recipient, subject, body, key, attachments=()):
        message = EmailMessage()
        message['From'] = self.settings['sender']
        message['To'] = recipient
        message['Subject'] = subject
        message['Message-ID'] = f'<ptb-feedback-{key}@phonetictoolbox.com>'
        message.set_content(body)
        for index,(mime,data) in enumerate(attachments,1):
            subtype={'image/png':'png','image/jpeg':'jpeg'}[mime]
            message.add_attachment(data,maintype='image',subtype=subtype,filename=f'feedback-{index}.{subtype}')
        with self.connection() as client:
            refused = client.send_message(message)
            if refused:
                raise OSError('Mail delivery refused')
