##rag llm using llama 
import os 
#loading the pdf and converting into vectors 
from llama_index import VecotrStoreIndex,SimpleDirectoryReader 
#will give all the response and their source 
from llama_index.response.pprint.utils import pprint_response 
#getting multiple responses 
from llama_index.retrievers import VectorIndexRetriever 
#changing my query engine 
from llama_index.query_engine import RetrieverQueryEngine 
from llama_index.indices.postprocessor import SimilarityPostProcessor


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")


#crearting hte meta data of the pdfs
documents = SimpleDirectoryReader("data").load_data()

#convert into index
index = VecotrStoreIndex.from_documents(documents,show_progess=True)



#querying the question from the index
query_engine = index.as_query_engine()

#indexs are used retriever  
retriever = VectorIndexRetriever(index=index,similarity_top_k=4)
postprocessor = SimilarityPostProcessor(similarity_cutoff=0.80)
 
query_engine = RetrieverQueryEngine(retriever=retriever,
                                    node_postprocessor=postprocessor)

response = query_engine.query("what is a transformer")
pprint_response(response,souce=True)

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

##runnig it on local computer 
# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are transformers?")
print(response)



