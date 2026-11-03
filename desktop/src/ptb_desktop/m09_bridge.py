"""Native picker grants; Python paths never come from task JSON or HTTP."""
from pathlib import Path
from uuid import uuid4
from .file_provider import checked_path,FileAccessError
from .native_i18n import native_error

MESSAGES={
    'm09_python_invalid':'请选择已有环境中的 python.exe。',
    'm09_gpu_unavailable':'这个环境无法使用 CUDA，请核对 GPU 版 PyTorch 和显卡驱动。',
    'm09_python_dependencies':'这个环境缺少推理依赖，需要 torch、numpy、scipy、librosa、matplotlib 和 huggingface-hub。',
    'm09_python_changed':'Python 环境已改变，请重新选择并检查兼容性。',
    'm09_vocoder_failed':'GPU 模型试运行失败，请检查环境兼容性或改用 CPU。',
    'm09_python_check_failed':'Python 环境检查未完成，请确认选择了可运行的 Python 环境。',
}

MESSAGE_KEYS={'m09_python_invalid': 'errors.m09.pythonInvalid', 'm09_gpu_unavailable': 'errors.m09.gpuUnavailable', 'm09_python_dependencies': 'errors.m09.pythonDependencies', 'm09_python_changed': 'errors.m09.pythonChanged', 'm09_vocoder_failed': 'errors.m09.vocoderTrialFailed', 'm09_python_check_failed': 'errors.m09.pythonCheckIncomplete'}


class M09Bridge:
    def __init__(self):self.grants={}

    def grant(self,path):
        path=Path(path).absolute();checked_path(path)
        from ptb_worker.m09_runtime import fingerprint
        try:identity=fingerprint(path)
        except ValueError as exc:raise native_error(FileAccessError(MESSAGES.get(str(exc),MESSAGES['m09_python_invalid'])),MESSAGE_KEYS.get(str(exc),MESSAGE_KEYS['m09_python_invalid'])) from None
        key=uuid4().hex;self.grants={key:identity}
        return dict(id=key,name=path.parent.name+' / '+path.name)

    def check(self,grant):
        from ptb_worker.m09_runtime import check_and_register,fingerprint
        identity=self.grants.get(grant)
        if identity is None:raise FileAccessError('请先选择已有 Python 环境。')
        try:
            if fingerprint(identity['path'])!=identity:raise ValueError('m09_python_changed')
            return check_and_register(identity['path'])
        except ValueError as exc:raise native_error(FileAccessError(MESSAGES.get(str(exc),MESSAGES['m09_python_check_failed'])),MESSAGE_KEYS.get(str(exc),MESSAGE_KEYS['m09_python_check_failed'])) from None
