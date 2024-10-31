FROM postgres:latest as database
ENV POSTGRES_DB postgres
COPY schema.sql /docker-entrypoint-initdb.d/