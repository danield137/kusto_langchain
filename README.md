# LangChain Kusto Vector Store

**⚠️ PRE-ALPHA VERSION - This package is in very early development and not recommended for production use.**

A LangChain vector store implementation for Azure Data Explorer (Kusto), Microsoft Fabric Eventhouse, and other Kusto-compatible databases.

## Current Status

This is a **very initial version** with **only retrieval capabilities**. Document storage functionality is not yet implemented.

## Features

- ✅ Retrieve vector embeddings from Azure Data Explorer (Kusto) or Microsoft Fabric Eventhouse
- ✅ Compatible with LangChain's vector store interface
- ✅ Similarity search with cosine similarity metric
- ❌ Document storage (not yet implemented)
- ❌ Batch operations (not yet implemented)

## Installation

```bash
pip install langchain-kusto
```

## Quick Start

```python
from langchain_kusto import KustoVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="your-openai-endpoint",
    azure_deployment="your-embedding-deployment",
    openai_api_version="2023-05-15"
)

# Initialize the vector store (retrieval only)
vector_store = KustoVectorStore(
    connection="https://your-cluster.kusto.windows.net",  # or KustoConnectionStringBuilder
    database="your_database",
    collection_name="your_table",
    embedding=embeddings,
    embedding_column="embedding_text",  # optional, defaults to "embedding"
    id_column="vector_id",              # optional, defaults to "id"
    content_column="doc_text"           # optional, defaults to "text"
)

# Search for similar documents (this requires pre-existing data in Kusto)
results = vector_store.similarity_search("your query text", k=5)
```

## Complete Example

See [demo.py](demo.py) for a complete working example using Azure OpenAI embeddings and a RAG (Retrieval-Augmented Generation) pipeline.

## Requirements

- Python >= 3.8
- Azure Data Explorer cluster or Microsoft Fabric Eventhouse with pre-existing vector data
- LangChain Core >= 0.1.0
- Azure authentication (DefaultAzureCredential)

## Data Prerequisites

Since this version only supports retrieval, you need to have your vector embeddings already stored in a Kusto table with the following structure:

```kql
.create table your_table (
    vector_id: string,
    doc_text: string,
    embedding_text: dynamic  // Array of float values representing the vector
    // ... other metadata columns
)
```

## Development Status

This package is currently in pre-alpha development. APIs may change significantly between versions. The current version only supports reading existing vector data from Kusto - document ingestion and storage capabilities will be added in future releases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
