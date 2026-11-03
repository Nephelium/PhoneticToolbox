"""Window-owned native captions and a small persistent language preference."""
import json
import os
from pathlib import Path
import re
import uuid
from .native_messages_data import NATIVE_MESSAGES

# Match complete, registered product errors at native dialog boundaries only.
_ERROR_KEYS={text:key for rows in reversed(tuple(NATIVE_MESSAGES.values())) for key,text in reversed(tuple(rows.items())) if key.startswith('errors.')}

def native_error(error,key,**values):
    """Attach a known display descriptor without changing exception type or text."""
    if key not in NATIVE_MESSAGES['zh-CN']:raise ValueError('Unknown native message key')
    if any(value is not None and not isinstance(value,(str,int,float,bool)) for value in values.values()):raise TypeError('Invalid native message value')
    error.display={'key':'native.'+key,'values':values}
    return error

def native_error_payload(error):
    result={'ok':False,'error':str(error)}
    display=getattr(error,'display',None)
    if isinstance(display,dict) and isinstance(display.get('key'),str) and display['key'].startswith('native.') and display['key'][7:] in NATIVE_MESSAGES['zh-CN']:
        result['display']=display
    return result


class NativeLocale:
    supported=tuple(NATIVE_MESSAGES)
    def __init__(self,preference_path=None):
        self.preference_path=Path(preference_path) if preference_path is not None else None
        self.locale='zh-CN'
        self._saved_code=None;self._saved_identity=None
        if self.preference_path is not None:
            try:
                file=self.preference_path
                identity=self._identity(file)
                if identity is not None:
                    value=json.loads(file.read_text('utf-8'))
                    if isinstance(value,dict) and isinstance(value.get('locale'),str) and value['locale'] in NATIVE_MESSAGES:
                        self.locale=value['locale'];self._saved_code=self.locale;self._saved_identity=identity
            except (OSError,ValueError):pass

    @staticmethod
    def _identity(file):
        status=file.stat()
        if file.is_symlink() or getattr(status,'st_file_attributes',0)&0x400 or status.st_size>16384:return None
        return status.st_size,status.st_mtime_ns,status.st_ctime_ns,status.st_ino

    def select(self,code):
        if not isinstance(code,str) or code not in NATIVE_MESSAGES:raise ValueError('unsupported_locale')
        self.locale=code
        if self.preference_path is None:return {'locale':code,'saved':True}
        target=self.preference_path
        try:
            if self._saved_code==code and self._saved_identity is not None and self._identity(target)==self._saved_identity:return {'locale':code,'saved':True}
        except OSError:pass
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_symlink() or target.exists() and getattr(target.stat(),'st_file_attributes',0)&0x400:raise OSError('Unsafe preference path')
            with temporary.open('x',encoding='utf-8',newline='\n') as stream:
                json.dump({'schema':'ptb-native-language/1','locale':code},stream);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,target)
            self._saved_code=code;self._saved_identity=self._identity(target)
            return {'locale':code,'saved':True}
        except OSError:return {'locale':code,'saved':False}
        finally:
            try:temporary.unlink(missing_ok=True)
            except OSError:pass

    def text(self,key,**values):
        text=NATIVE_MESSAGES[self.locale].get(key,NATIVE_MESSAGES['zh-CN'].get(key,key))
        return re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}',lambda match:str(values.get(match[1],match[0])),text)

    def error(self,value):
        text=str(value)
        key=_ERROR_KEYS.get(text)
        return self.text(key) if key else text

    def startup_failure(self,error):
        key=_ERROR_KEYS.get(str(error))
        reason=self.text(key) if key else self.text('startup.integrityOrAccessFailure')
        return self.text('startup.failureTitle'),self.text('startup.failureBody',reason=reason)


_default=NativeLocale()


def native_text(window,key,**values):
    return getattr(window,'native_locale',_default).text(key,**values)
