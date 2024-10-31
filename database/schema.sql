CREATE TABLE scores (
    id serial primary key,
    player text not null,
    score integer not null
);

CREATE USER web WITH PASSWORD 'web';

GRANT SELECT, INSERT ON scores TO web;
GRANT USAGE ON SEQUENCE scores_id_seq TO web;