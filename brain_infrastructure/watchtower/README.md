# Watchtower
Watchtower is our monitor which ensures that all running container, started with the `com.centurylinklabs.watchtower.enable=true`flag are on the latest version. 

To start the service you first need to create a config.json containing the credentials to log into the container registry. Therefor usually a repo token with an arbitrary username works fine.
The easiest way is to create a folder inside the home directory of the user running the docker container `~/.docker/config.json` and add the following content:
```json
{
    "auths": {
            "registry.code.fbi.h-da.de": {
                    "auth": ""
            }
    }
}
```
After that run `docker login registry.code.fbi.h-da.de` and enter the arbitrary username and repo token.
Afterwards the _config.json_ value contains the base64 encoded credentials. This _config.json_ then needs to be mounted into the watchtower container inside the _compose.yml_ file
After making the needed adjustments to _compose.yml_ simply run `docker compose up -d`.
