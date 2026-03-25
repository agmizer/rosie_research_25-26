from Chunking import Chunking
from EmbeddingsClass import Embeddings
import faiss
import numpy as np
import os


class RAG:
    """
        This class is used to make a full RAG section for the pipeline
    """

    def __init__(self):
        self.chunks = []
        self.index = faiss.IndexFlatL2(384)
        self.embedder = Embeddings()
        self.chunker = Chunking()

    def add_data(self, data_path, data_name):
        """
        This method takes in some data, determines its type, calls the correct chunking method, then adds the chunked
        JSON result to chunks and the embedded version of the chunks to embeddings
        """

        if not data_name:
            data_name = os.path.splitext(os.path.basename(data_path))[0]

        if data_path.endswith(".pdf"):
            docs = self.chunker.extract_text_from_pdf(data_path)
        else:
            docs = self.chunker.extract_all_content(data_path)

        json_chunks = self.chunker.chunk_with_langchain(docs, data_name)

        texts = [chunk.page_content for chunk in json_chunks]

        embeddings = self.embedder.embed_text(texts)

        embeddings = embeddings.cpu().numpy().astype("float32")

        #Normalize (for cosine similarity)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        for chunk in json_chunks:
            chunk.metadata["dataset"] = data_name
            self.chunks.append(chunk)



    def get_data(self, query, k):
        """
            Takes the query, and find the best match to it in the stored data. Results the full JSON result stored in the 
            chunk structure for each k number of chunks that match. 
        """

        query_vector = self.embedder.embed_text(query)
        query_vector = np.array(query_vector).astype("float32")


        D, I = self.index.search(query_vector, k)
        return [self.chunks[i] for i in I[0]]