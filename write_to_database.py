import pandas as pd
import numpy as np
import sqlite3

chunksize = 1000000
names = ["userid", "offerid", "countrycode", "category", "merchant", "utcdate", "rating"]


nb_users = 291485
nb_offers = 2158859

# CONNECT TO DB
conn = sqlite3.connect('kassandr.db')
c = conn.cursor()

# DROP users TABLE
c.execute("DROP TABLE users")
# CREATE users TABLE
c.execute("CREATE TABLE users (id int primary key,dumb int);")
# CREATE 291485 users
for i in range(0, nb_users):    
    c.execute("INSERT INTO users (id, dumb) VALUES(?,?)", [i,i]);
for row in c.execute("SELECT * from users limit 10"):
    print(row)

# CREATE offers TABLE
c.execute("CREATE TABLE offers (id int primary key,dump int);")
# DROP offers TABLE
c.execute("DROP TABLE offers")
# CREATE 2158860 offers
for i in range(0, nb_offers):
    c.execute("INSERT INTO offers (id, dump) VALUES(?, ?)", [i, i])
for row in c.execute("SELECT * FROM offers LIMIT 1"):
    print(row)


# DROP ratings TABLE
c.execute("DROP TABLE ratings;")
# CREATE RATINGS TABLE
c.execute("CREATE TABLE ratings (user_id int,offer_id int, rating int, FOREIGN KEY(user_id) REFERENCES users(id)" + 
                                 ",FOREIGN KEY(offer_id) REFERENCES offers(id));")
# FILL IN RATINGS
for i in range(1, 17):
    print("Inserting file: " + str(i))
    for training_chunk in pd.read_csv('train/train' + str(i) + '_replaced.csv', chunksize=chunksize, names=names):
        training_chunk = np.array(training_chunk, dtype='str')
        for i in range(2, len(training_chunk[:, 0])):
            try:
                c.execute("INSERT INTO ratings (user_id, offer_id, rating)  VALUES(?, ?, ?)", 
                      [int(training_chunk[:, 0][i]), int(training_chunk[:, 1][i]), int(training_chunk[:, 6][i])])
            except ValueError:
                pass
for row in c.execute("SELECT COUNT(*) FROM offers"):
    print(row)
    
for row in c.execute("SELECT MIN(offer_id) FROM ratings"):
    print (row)
for row in c.execute("SELECT COUNT(*) FROM users"):
    print(row)
    
c.execute("INSERT INTO ratings (user_id, offer_id, rating) VALUES (0,2158860,0)")
    
    
    
# CREATE ratings_test TABLE 
c.execute("CREATE TABLE ratings_test(user_id int,offer_id int, rating int, FOREIGN KEY(user_id) REFERENCES users(id)" + 
                                 ",FOREIGN KEY(offer_id) REFERENCES offers(id));")
# FILL IN TEST RATINGS
for i in range(1, 3):
    print("Inserting file: " + str(i))
    for training_chunk in pd.read_csv('test/test' + str(i) + '_replaced.csv', chunksize=chunksize, names=names):
        training_chunk = np.array(training_chunk, dtype='str')
        for i in range(0, len(training_chunk[:, 0])):
            try:
                c.execute("INSERT INTO ratings_test (user_id, offer_id, rating)  VALUES(?, ?, ?)", 
                      [int(training_chunk[:, 0][i]), int(training_chunk[:, 1][i]), int(training_chunk[:, 6][i])])
            except ValueError:
                pass

conn.commit()

c.close()  
