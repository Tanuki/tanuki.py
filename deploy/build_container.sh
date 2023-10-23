#! /bin/bash
docker build \
    -f Dockerfile \
    --target production \
    -t monkey-patch-apps \
    .