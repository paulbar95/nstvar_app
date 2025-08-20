# Getting Started

## Start the containers
### pm25-service
Move to the service's directory ``C:\Users\paulb\Documents\Projekte\Uni\UAS_Master\Master\pm25_service`` and run 
``docker compose up -d
``.

### minio
Move to the service's directory ``C:\Users\paulb\Documents\Projekte\Uni\UAS_Master\Master\minio`` and run 
`docker compose up -d
``.[^1]

[^1]: Alternatively open the ``dockerdesktop`` app and start the containers manually.

## Start local instances
### nstvar (Backend/SpringBoot)
Move to ``C:\Users\paulb\Documents\Projekte\Uni\UAS_Master\Master\nstvar`` and run the project via the selected IDE 
(e.g. Intellij).

### postgreSQL (RDB)
Open the ``pgAdmin 4`` app, start the ``PostgreSQL 14`` server. There you will find the ``nstvar`` database.
