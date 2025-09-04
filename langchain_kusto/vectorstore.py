from __future__ import annotations
from typing import Any, List, Optional, Dict, Iterable, TypeVar
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from azure.identity import DefaultAzureCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder, ClientRequestProperties
import json

VST = TypeVar("VST", bound="KustoVectorStore")

class KustoVectorStore(VectorStore):
    """Vector store implementation for Azure Data Explorer (Kusto)."""
    
    def __init__(
        self,
        cluster_uri: str,
        database: str,
        collection_name: str,
        embedding: Optional[Embeddings] = None,
        embedding_column: str = "embedding",
        id_column: str = "id",
        content_column: str = "text",
        **kwargs: Any
    ):
        """Initialize KustoVectorStore.
        
        Args:
            cluster_uri: Kusto cluster URI
            database: Database name
            collection_name: Table name for the vector collection
            embedding: Embeddings instance for query embedding
            **kwargs: Additional arguments
        """
        self.cluster_uri = cluster_uri
        self.database = database
        self.collection_name = collection_name
        self._embedding = embedding
        self.embedding_column = embedding_column
        self.id_column = id_column
        self.content_column = content_column

        # Initialize Kusto client
        credential = DefaultAzureCredential(
            exclude_workload_identity_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_interactive_browser_credential=False
        )
        kcsb = KustoConnectionStringBuilder.with_azure_token_credential(cluster_uri, credential)
        self._client = KustoClient(kcsb)
    
    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access to embeddings instance."""
        return self._embedding
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        if self._embedding is None:
            raise ValueError("Embeddings instance required for similarity search")
        
        # Generate query embedding
        query_embedding = self._embedding.embed_query(query)
        
        # Build KQL query
        query_vector_str = json.dumps(query_embedding)
        kql_query = f"""
        let q = dynamic('{query_vector_str}');
        {self.collection_name}
        | extend sim = series_cosine_similarity({self.embedding_column}, q)
        | top {k} by sim desc
        """
        
        # Execute query
        response = self._client.execute(self.database, kql_query)
        
        # Convert results to Documents
        documents = []
        metadata = {}
        for row in response.primary_results[0]:
            d = row.to_dict()
            text = d[self.content_column]
            _id = d[self.id_column] if self.id_column in d else None
            metadata = {k: v for k, v in d.items() if k not in [self.content_column, self.embedding_column, "sim"]}
            documents.append(Document(page_content=text, metadata=metadata, id=_id))
        
        return documents

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        **kwargs: Any
    ) -> "KustoVectorStore":
        """Not implemented - Kusto is immutable."""
        raise NotImplementedError("Not implemented yet.")
