#! /bin/bash
docker run \
    --rm \
    -it \
    -p 8000:8000 \
    --mount type=bind,source="$(pwd)"/apps,target=/app \
    --env-file .env \
    monkey-patch-apps \
    bash