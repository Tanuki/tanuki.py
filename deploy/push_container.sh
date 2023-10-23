#! /bin/bash

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 238337697524.dkr.ecr.us-east-2.amazonaws.com

docker tag monkey-patch-apps:latest 238337697524.dkr.ecr.us-east-2.amazonaws.com/monkey-patch-apps:latest

docker push 238337697524.dkr.ecr.us-east-2.amazonaws.com/monkey-patch-apps:latest
