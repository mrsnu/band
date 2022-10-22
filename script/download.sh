IMAGE_NAME=$(docker ps | grep vsc-band | awk 'NF>1{print $NF}')
docker cp $IMAGE_NAME://workspaces/band/$1 $1

