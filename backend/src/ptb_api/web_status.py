"""Cached aggregate load for signed-in web users; no process or account details."""
from pathlib import Path
from threading import Lock
import time
from .models import WireModel


class ServerLoadView(WireModel):
    running_tasks: int
    waiting_tasks: int
    task_slots: int
    cpu_percent: float | None = None
    memory_percent: float | None = None
    measured_at: float


def host_counters(proc=Path('/proc')):
    # No scientific dependencies or blocking CPU measurement. Unsupported hosts
    # and unavailable counters remain unknown, never a fabricated zero.
    cpu=memory=None
    try:
        parts=(proc/'stat').read_text().splitlines()[0].split()
        if parts[0]=='cpu':
            ticks=[int(x) for x in parts[1:9]]
            if len(ticks)==8:cpu=(sum(ticks),ticks[3]+ticks[4])
    except (OSError,ValueError,IndexError):pass
    try:
        values={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in (proc/'meminfo').read_text().splitlines())}
        total,available=values['MemTotal'],values['MemAvailable']
        if total>0 and 0<=available<=total:memory=round(100*(total-available)/total,1)
    except (OSError,ValueError,KeyError,IndexError):pass
    return cpu,memory


class PublicLoad:
    """One shared count read per second; host sampling only on detail requests."""
    def __init__(self,service,jobs,*,clock=time.monotonic,probe=host_counters):
        self.service,self.jobs,self.clock,self.probe=service,jobs,clock,probe
        self.lock=Lock();self.counts=None;self.counts_until=0.;self.cached=None;self.until=0.;self.previous=None;self.previous_at=None

    def sample(self, *, details=True):
        if self.jobs is None:return None
        with self.lock:
            now=self.clock()
            if self.counts is None or now>=self.counts_until:
                with self.service._read() as c:
                    counts=c.execute("""SELECT count(*) FILTER (WHERE state IN ('running','cancel_requested')) AS running,
                    count(*) FILTER (WHERE state='queued') AS waiting,
                    (SELECT count(*) FROM ptb_jobs.acoustic_batch_items i
                     JOIN ptb_jobs.acoustic_batches b ON b.id=i.batch_id
                     WHERE b.closed_at IS NULL AND NOT b.cancel_requested AND i.child_job_id IS NULL) AS batch
                    FROM ptb_jobs.jobs
                    WHERE state IN ('running','cancel_requested','queued')""").fetchone()
                    measured=self.service.now(c)
                self.counts=ServerLoadView(running_tasks=counts['running'],waiting_tasks=counts['waiting']+counts['batch'],
                    task_slots=self.jobs.max_running,measured_at=measured)
                self.counts_until=now+1
            if not details:return self.counts
            if self.cached is not None and now<self.until:
                keys=('running_tasks','waiting_tasks','task_slots')
                if all(getattr(self.cached,k)==getattr(self.counts,k) for k in keys):return self.cached
                return self.cached.model_copy(update={k:getattr(self.counts,k) for k in keys})
            current,memory=self.probe();cpu=None
            if current is not None and self.previous is not None and self.previous_at is not None and now-self.previous_at<=2.5:
                total,idle=current[0]-self.previous[0],current[1]-self.previous[1]
                if total>0 and 0<=idle<=total:cpu=round(100*(total-idle)/total,1)
            self.previous,self.previous_at=current,now
            self.cached=self.counts.model_copy(update=dict(cpu_percent=cpu,memory_percent=memory))
            self.until=now+1
            return self.cached
