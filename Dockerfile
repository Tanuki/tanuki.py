# -- BASE --
FROM python:3.11 as base

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    curl \
    jq \
    vim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY /src/ /src/
COPY /apps/ /app/

# -- LOCAL --
FROM base as local-dev

# -- CLOUD DEPLOY --
FROM base as production

ENTRYPOINT ["/bin/sh", "-c" , "./entrypoint.sh" ]