"""P05 test double only: not SQL, persistence, isolation or concurrency evidence."""
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from ptb_api.account_store import PASSWORDS

class MemoryAccountStore:
    def __init__(self):
        self.users, self.sessions, self.projects, self.attempts = {}, {}, {}, {}
    def create_user(self, username, password):
        self.users[username] = dict(id=str(uuid4()),username=username,password_hash=PASSWORDS.hash(password),active=True)
    def get_user(self, username):
        return self.users.get(username)
    def allow_login(self, username, ip):
        keys = [('user:'+username,10),('ip:'+ip,50)]
        for key,limit in keys:
            self.attempts[key] = self.attempts.get(key,0)+1
        return all(self.attempts[key]<=limit for key,limit in keys)
    def issue_session(self,user,token_hash,csrf,expires,old_hash):
        self.revoke(old_hash)
        self.sessions[token_hash] = dict(id=user['id'],username=user['username'],csrf_token=csrf,expires_at=expires,revoked=False)
        return True
    def session(self,token_hash):
        item = self.sessions.get(token_hash)
        if item and not item['revoked'] and item['expires_at']>datetime.now(timezone.utc) and self.users[item['username']]['active']:
            return item
    def revoke(self,token_hash):
        if token_hash in self.sessions:self.sessions[token_hash]['revoked']=True
    def list_projects(self,owner):
        return [{k:v for k,v in p.items() if k!='owner'} for p in self.projects.values() if p['owner']==owner]
    def create_project(self,owner,name):
        if len(self.list_projects(owner))>=100:return None
        p=dict(id=str(uuid4()),name=name,created_at=datetime.now(timezone.utc),owner=owner)
        self.projects[p['id']]=p
        return self.get_project(owner,p['id'])
    def get_project(self,owner,project_id):
        p=self.projects.get(project_id)
        if p and p['owner']==owner:return {k:v for k,v in p.items() if k!='owner'}
    def rename_project(self,owner,project_id,name):
        if self.get_project(owner,project_id):
            self.projects[project_id]['name']=name
            return self.get_project(owner,project_id)
