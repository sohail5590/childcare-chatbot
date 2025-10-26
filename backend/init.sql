-- Create the database if it doesn't already exist.
CREATE DATABASE IF NOT EXISTS moviedb;

-- Switch to using the 'moviedb' database.
USE moviedb;

-- Create the 'movies' table with a schema matching the CSV file.
CREATE TABLE movies (
    Id INT  PRIMARY KEY COMMENT 'Unique identifier for each movie record. This also serves a Movie Id',
        
    Movie_Name VARCHAR(255) NOT NULL COMMENT 'The official title of the movie. This field cannot be null.',
        
    Year_Of_Release INT COMMENT 'The calendar year in which the movie was first released to the public.',
        
    Watch_Time VARCHAR(50) COMMENT 'The total runtime or duration of the movie, typically specified in minutes (e.g., "148 min").',
        
    Movie_Rating DECIMAL(3, 1) COMMENT 'The user or critic rating for the movie, typically on a scale from 1.0 to 10.0.',
        
    Metascore_of_Movie INT COMMENT 'The Metascore from Metacritic, a weighted average of reviews from top critics, typically ranging from 0 to 100.',
        
    Votes INT COMMENT 'The total number of user votes or ratings submitted for the movie.',
        
    Gross VARCHAR(50) COMMENT 'The total gross box office revenue earned by the movie. Stored as a string to accommodate currency symbols and abbreviations (e.g., "$150.5M").',
        
    Description TEXT COMMENT 'A brief text summary or synopsis of the movie''s plot.'
);

-- Create indexes on columns likely to be used in queries to improve performance.
CREATE INDEX idx_year ON movies (Year_Of_Release);
CREATE INDEX idx_rating ON movies (Movie_Rating);
CREATE INDEX idx_metascore ON movies (Metascore_of_Movie);