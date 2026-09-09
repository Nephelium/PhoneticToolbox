"""One lifecycle policy for PostgreSQL and local SQLite adapters."""
TERMINAL = frozenset({'succeeded','failed','cancelled','interrupted'})
ACTIVE = frozenset({'running','cancel_requested'})


def cancel_state(state):
    return {'queued':'cancelled','running':'cancel_requested'}.get(state,state)


def fenced(row, worker_id, generation, now):
    return (row['state'] in ACTIVE and row['worker_id'] == worker_id
            and row['generation'] == generation and row['lease_until'] > now and row['deadline'] > now)
