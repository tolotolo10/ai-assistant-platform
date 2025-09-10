import os, pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"], 
)

print("Existing indexes:", pinecone.list_indexes())
# CAREFUL: delete ones you don't need
to_delete = ["ai-assistant-index", "old-index-1"]  
for name in to_delete:
    if name in pinecone.list_indexes():
        print("Deleting", name)
        pinecone.delete_index(name)
print("Done. Remaining:", pinecone.list_indexes())
