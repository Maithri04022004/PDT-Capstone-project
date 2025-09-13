
import pymongo

# Replace with your Atlas connection string URI
uri = "mongodb+srv://Maithri:Maithri_2025@cluster0.vefa5u1.mongodb.net/"

client = pymongo.MongoClient(uri)

db = client['PDT_DB']       # Database name
collection = db['Collection_01']      # Collection name

# Fetch and print 5 most recent records
for doc in collection.find().sort("timestamp", -1).limit(5):
    print(doc)

client.close()

