import chromadb
client = chromadb.PersistentClient(path='./data/data/vector_db')
collection = client.get_collection('fds_imagens')

# Obtener un chunk de imagen
sample = collection.get(limit=1, include=['metadatas'])
print("Metadata de chunk imagen:")
print(sample['metadatas'][0])