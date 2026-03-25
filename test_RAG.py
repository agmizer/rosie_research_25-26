from RAG import RAG
from pathlib import Path
import time

K = 3

def test_rag_with_project_report():
    rag = RAG()

    # Step 1: Add your PDF
    rag.add_data("/home/ad.msoe.edu/sterritts/ROSIE_Tutor/rosie_research_25-26/pdfs/Samantha Sterritt - Paper Review 6.pdf", None)

    # Step 2: Make a query (adjust based on your PDF content)
    query = "What is the main objective of the project?"

    # Step 3: Retrieve top-k chunks
    results = rag.get_data(query, k=K)

    # Step 4: Assertions (basic correctness checks)
    assert len(results) == K, "Should return k results"

    for doc in results:
        # Check structure
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")

        # Check metadata fields exist
        assert "source" in doc.metadata
        assert "page" in doc.metadata

        # Optional: check your custom metadata
        assert doc.metadata.get("dataset") == "Samantha Sterritt - Paper Review 6"

        # Ensure text is not empty
        assert len(doc.page_content.strip()) > 0

    # Step 5: Print results (for debugging / validation)
    print("\nTop Results:\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Page: {doc.metadata['page']}")
        print(f"Source: {doc.metadata['dataset']}")
        print(doc.page_content[:300])  # first 300 chars
        print("-" * 50)


global_rag = RAG()

def test_fill_rag_all_data():
    folder = Path("RAGInitialLoadData")

    # Load all files into RAG
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            global_rag.add_data(str(file_path), None)

    # --- Assertions ---
    # Ensure data was actually added
    assert len(global_rag.chunks) > 0, "No chunks were added to the RAG system"

    # Ensure FAISS index has vectors
    assert global_rag.index.ntotal > 0, "FAISS index is empty"

    # Try a sample query to confirm retrieval works
    test_query = "test query"
    results = global_rag.get_data(test_query, k=3)

    assert isinstance(results, list), "get_data should return a list"
    assert len(results) > 0, "No results returned from query"
    print("done")
    

def test_query_global():
    query = "Explain Nearest Neighbors"
    results = global_rag.get_data(query, k=30)
    print("\nTop Results:\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Page: {doc.metadata['page']}")
        print(f"Source: {doc.metadata['dataset']}")
        print(doc.page_content[:300])  # first 300 chars
        print("-" * 50)
    

#test_rag_with_project_report()

start_time = time.time()
test_fill_rag_all_data()
end_time = time.time()
elapsed_time = end_time - start_time
print("FILL GLOBAL RAG TIME: ", elapsed_time)

start_time = time.time()
test_query_global()
end_time = time.time()
elapsed_time = end_time - start_time
print("QUERY GLOBAL RAG TIME: ", elapsed_time)