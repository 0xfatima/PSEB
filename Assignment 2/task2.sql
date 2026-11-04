

CREATE DATABASE library;

-- Connect to the library database before running this section.

CREATE TABLE books (
    id INT PRIMARY KEY,
    title VARCHAR(255),
    author VARCHAR(255),
    year INT,
    price FLOAT
);
INSERT INTO books (id, title, author, year, price)
VALUES
(1, 'The Alchemist', 'Paulo Coelho', 1988, 750),
(2, 'Atomic Habits', 'James Clear', 2018, 280),
(3, '1984', 'George Orwell', 1949, 1000),
(4, 'The Great Gatsby', 'F. Scott Fitzgerald', 1925, 350),
(5, 'Pride and Prejudice', 'Jane Austen', 1813, 200),
(6, 'The Hobbit', 'J.R.R. Tolkien', 1937, 600),
(7, 'To Kill a Mockingbird', 'Harper Lee', 1960, 400),
(8, 'Harry Potter and the Philosophers Stone', 'J.K. Rowling', 1997, 100);

SELECT title FROM books WHERE price > 500;
select title, price from books order by price desc;
select count(*) from books;
update books set price = 300  where title ='The Alchemist';
