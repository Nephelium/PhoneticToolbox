"""Check the authorized SMTP source without executing it or sending email."""
import argparse
import sys
import smtplib
from workbench_source import bind_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    args=parser.parse_args()
    bind_sources()
    from ptb_api.feedback_mail import settings_from_script,SMTPMailer
    try:
        SMTPMailer(settings_from_script(args.source)).check()
    except smtplib.SMTPAuthenticationError as exc:
        print('SMTP_AUTH_REJECTED: status '+str(exc.smtp_code)+'; no private details logged.')
        return 1
    except Exception as exc:
        print('SMTP_CHECK_FAILED: '+type(exc).__name__+'; no private details logged.')
        return 1
    print('SMTP_CHECK_OK: TLS and authentication verified; no email sent.')
    return 0


if __name__=='__main__':raise SystemExit(main())
