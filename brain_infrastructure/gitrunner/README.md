# Gitrunner
This repo contans the configuraiton of all gitlab runners used for the project.
For each runner the corresponding .env files needs to be created containing the following information:
```
RUNNER_TAG_LIST=<arbitrary name of the runner>
REGISTRATION_TOKEN="<registration_token>"
```
The registration Token can be found inside gitlab inside the corresponding repository under "Settings -> CI/CD -> Runners -> Project Runners". Next to **New project runner** click the three dots and copy the _Registration token_.

To start the runner simply run `docker-compose up --build -d`.