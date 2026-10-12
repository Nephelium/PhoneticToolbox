"""Use installed Android tools; keep the private signing key outside source."""
from pathlib import Path
import os
import secrets
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/src'))
from ptb_desktop.feedback import protected

def main():
    jdk=Path(r'D:\Android Studio\jbr')
    sdk=Path(os.environ['LOCALAPPDATA'])/'Android/Sdk'
    gradles=list((Path.home()/'.gradle/wrapper/dists/gradle-8.14.3-all').glob('*/gradle-8.14.3/bin/gradle.bat'))
    if len(gradles)!=1 or not (jdk/'bin/keytool.exe').is_file():raise RuntimeError('Installed Android build tools are unavailable')
    private=Path(os.environ['LOCALAPPDATA'])/'PhoneticToolbox/v3/feedback-manager/signing'
    private.mkdir(parents=True,exist_ok=True)
    password_file=private/'signing-password.bin'
    keystore=private/'feedback-release.jks'
    if keystore.exists() and not password_file.exists():raise RuntimeError('Preserve the existing key and recover its signing password')
    if password_file.exists():password=protected(password_file.read_bytes(),True).decode()
    else:
        password=secrets.token_urlsafe(36)
        password_file.write_bytes(protected(password.encode()))
    env={**os.environ,'JAVA_HOME':str(jdk),'ANDROID_HOME':str(sdk),'PTB_FEEDBACK_KEYSTORE':str(keystore),'PTB_FEEDBACK_STORE_PASSWORD':password}
    if not keystore.exists():
        subprocess.run([str(jdk/'bin/keytool.exe'),'-genkeypair','-keystore',str(keystore),'-storepass:env','PTB_FEEDBACK_STORE_PASSWORD','-keypass:env','PTB_FEEDBACK_STORE_PASSWORD','-alias','feedback','-keyalg','RSA','-keysize','3072','-validity','10000','-dname','CN=PhoneticToolbox Feedback Inbox'],env=env,check=True)
    tasks=['assembleRelease']+(['lintRelease'] if '--lint' in sys.argv else [])
    subprocess.run([str(gradles[0]),'--offline','--no-daemon','--console=plain',*tasks],env=env,cwd=ROOT/'tools/feedback-manager/android',check=True)
    apk=ROOT/'output/build-feedback-manager/android-app/outputs/apk/release/app-release.apk'
    destination=ROOT/'output/feedback-manager/1.0.1/反馈收件箱.apk'
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(apk,destination)
    print('Android release APK created')
if __name__=='__main__':main()
