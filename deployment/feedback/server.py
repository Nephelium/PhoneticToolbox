"""Feedback-only deployment entry; same API implementation as unified backend."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check-smtp',action='store_true')
    options=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'backend/src'))
    from ptb_api.feedback import FeedbackService,create_feedback_app
    from ptb_api.feedback_mail import SMTPMailer,settings_from_script
    config=json.loads(options.config.read_text(encoding='utf-8-sig'))
    smtp=config.get('smtp')
    if config.get('smtpSource'):
        smtp=settings_from_script(config['smtpSource'])
    mailer=SMTPMailer(smtp) if smtp else None
    if options.check_smtp:
        if mailer is None:
            raise ValueError('SMTP configuration unavailable')
        mailer.check()
        print('SMTP authentication verified; no email was sent.')
        return
    service=FeedbackService(config['database'],secret=config['secret'],admin_token=config['adminToken'],
                            mailer=mailer,daily_hour=config.get('dailyHour',9),allow_unverified=config.get('allowUnverified',False))
    if config.get('webOperations'):
        from ptb_api.feedback_web_ops import WebOpsGateway
        service.web_operations=WebOpsGateway(config['webOperations'])
    import uvicorn
    uvicorn.run(create_feedback_app(service),host='127.0.0.1',port=config.get('port',18766),access_log=False,log_level='warning',proxy_headers=False,
                ssl_certfile=config.get('tls_certfile'),ssl_keyfile=config.get('tls_keyfile'))


if __name__=='__main__':
    try:
        main()
    except Exception:
        print('Feedback startup or SMTP check failed; inspect private settings locally.',file=sys.stderr)
        raise SystemExit(1) from None
