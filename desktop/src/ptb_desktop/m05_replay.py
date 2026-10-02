"""Bounded reads of full M05 frames, including results made before R1."""
import json
from .file_provider import FileAccessError


class ReplayReader:
    def __init__(self):
        self.indices = {}

    def read(self, service, job, start):
        if type(start) is not int or not 0 <= start <= 300_000 or start % 90:
            raise FileAccessError('回放帧范围不正确。')
        if job['state'] != 'succeeded' or job['operation'] != 'lip_analysis':
            raise FileAccessError('完整唇形结果尚不可用。')
        item = next((f for f in job['result_manifest']['files'] if f['name'] == 'frames.jsonl'), None)
        if not item or item['size_bytes'] > 512_000_000:
            raise FileAccessError('完整逐帧记录不可用。')
        key = (item['id'], item['sha256'])
        if key not in self.indices:
            if len(self.indices) >= 8:
                self.indices.pop(next(iter(self.indices)))
            self.indices[key] = {0: 0}
        marks = self.indices[key]
        index = max(i for i in marks if i <= start)
        position = marks[index]
        pending = b''
        line_start = position
        result = []
        while position < item['size_bytes'] or pending:
            if b'\n' not in pending and position < item['size_bytes']:
                size = min(1_048_576, item['size_bytes'] - position)
                raw = service.binary(f'/api/v1/jobs/local-results/{item["id"]}?offset={position}&size={size}')
                if len(raw) != size:
                    raise FileAccessError('逐帧结果读取不完整。')
                pending += raw
                position += size
            line, sep, rest = pending.partition(b'\n')
            if not sep and position < item['size_bytes']:
                if len(pending) > 200_000:
                    raise FileAccessError('逐帧记录超过预算。')
                continue
            if not line or len(line) > 200_000:
                raise FileAccessError('逐帧记录格式不正确。')
            pending = rest
            if index % 90 == 0:
                marks[index] = line_start
            if index >= start:
                row = json.loads(line)
                result.append({k: row[k] for k in ('index', 'time_s', 'detected', 'points', 'metrics', 'width', 'height') if k in row})
            index += 1
            line_start += len(line) + len(sep)
            if len(result) == 90:
                marks[index] = line_start
                break
        return dict(rows=result, start=start, complete=line_start == item['size_bytes'])
