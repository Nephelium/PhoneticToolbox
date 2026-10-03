#!/bin/sh
set -eu
/usr/sbin/nginx -t -q
/usr/bin/systemctl reload nginx
