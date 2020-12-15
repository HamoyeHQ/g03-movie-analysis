#!/bin/sh

image_name=gcr.io/$PROJECT_ID/kubeflow/moviesuccess/train   # Specify the image name here
image_tag=latest

full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"

docker build -t "${full_image_name}" .
docker push "$full_image_name"

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name_1}"

# docker inspect command outputs a hash link to the hosted image. This should be used when defining components
# Small script that runs docker build and docker push