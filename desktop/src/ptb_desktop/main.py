import json

from .local_service import LocalService


def main():
    with LocalService() as service:
        health = service.get('/api/v1/health')
    if service.exit_code != 0:
        raise RuntimeError('Local service did not exit cleanly')
    print(json.dumps({'health': health, 'service_exit_code': service.exit_code}))


if __name__ == '__main__':
    main()
