from RAG import RAG

K = 3

def test_rag_with_project_report():
    rag = RAG()

    # Step 1: Add your PDF
    rag.add_data("/home/ad.msoe.edu/sterritts/ROSIE_Tutor/rosie_research_25-26/pdfs/Samantha Sterritt - Paper Review 6.pdf", "Samantha Sterritt - Paper Review 6")

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

test_rag_with_project_report()