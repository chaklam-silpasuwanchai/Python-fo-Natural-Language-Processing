# Retrieval Augmented Generation (RAG)

General-purpose language models can be fine-tuned for tasks like sentiment analysis and named entity recognition, which don’t require extra background knowledge.  

For more complex, knowledge-intensive tasks, integrating external knowledge sources can enhance factual accuracy, improve reliability, and reduce hallucinations.  

To tackle this, Meta AI introduced **Retrieval-Augmented Generation (RAG)**, which combines information retrieval with text generation. Instead of relying solely on a model’s static knowledge, RAG retrieves relevant documents (e.g., from Wikipedia) and uses them as context when generating responses. This allows models to stay updated without requiring frequent retraining.  

[Lewis et al. (2021)](https://arxiv.org/pdf/2005.11401) proposed a fine-tuning approach for RAG, where a pre-trained seq2seq model serves as parametric memory, and a dense vector index of Wikipedia functions as non-parametric memory, accessed via a neural retriever.

<img src="./figures/rag-lewis.png" >

## Existing RAG Techniques
Here are the details of all the Advanced RAG techniques covered in this repository.

| Techniques | Description |
| --- | --- |
| Native RAG | Combines retrieved data with LLMs for simple and effective responses. |
| Hybrid RAG | Combines vector search and traditional methods like BM25 for better information retrieval. |
| Hyde RAG | Creates hypothetical document embeddings to find relevant information for a query. |
| Parent Document Retriever | Breaks large documents into small parts and retrieves the full document if a part matches the query. |
| RAG fusion | Generates sub-queries, ranks documents with Reciprocal Rank Fusion, and uses top results for accurate responses. |
| Contextual RAG | Compresses retrieved documents to keep only relevant details for concise and accurate responses. |
| Rewrite Retrieve Read (RRR) | Improves query, retrieves better data, and generates accurate answers. |
| Unstructured RAG | This method designed to handle documents that combine text, tables, and images. |