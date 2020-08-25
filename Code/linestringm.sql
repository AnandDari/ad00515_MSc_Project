ALTER TABLE ubertest
ADD COLUMN geom2 geometry;

update ubertest set geom2 = ST_AddMeasure(linestring,start_time,end_time) 
