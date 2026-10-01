"""
NOTE: This vector database integration is community-supported and maintained on a best-effort basis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import grpc
from open_webui.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_PREFIX,
    QDRANT_GRPC_PORT,
    QDRANT_HNSW_M,
    QDRANT_ON_DISK,
    QDRANT_PREFER_GRPC,
    QDRANT_TIMEOUT,
    QDRANT_URI,
)
from open_webui.retrieval.vector.main import (
    GetResult,
    SearchResult,
    VectorDBBase,
    VectorItem,
)
from open_webui.retrieval.vector.utils import iter_filter_conditions, process_metadata
from qdrant_client import QdrantClient as Qclient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import models

SCROLL_PAGE_SIZE = 1000
TENANT_ID_FIELD = 'tenant_id'

log = logging.getLogger(__name__)


def _tenant_filter(tenant_id: str) -> models.FieldCondition:
    return models.FieldCondition(key=TENANT_ID_FIELD, match=models.MatchValue(value=tenant_id))


def _metadata_filter(key: str, op: str, value: Any) -> models.FieldCondition:
    match = models.MatchAny(any=value) if op == '$in' else models.MatchValue(value=value)
    return models.FieldCondition(key=f'metadata.{key}', match=match)


class QdrantClient(VectorDBBase):
    def __init__(self):
        self.collection_prefix = QDRANT_COLLECTION_PREFIX
        self.QDRANT_URI = QDRANT_URI
        self.QDRANT_API_KEY = QDRANT_API_KEY
        self.QDRANT_ON_DISK = QDRANT_ON_DISK
        self.PREFER_GRPC = QDRANT_PREFER_GRPC
        self.GRPC_PORT = QDRANT_GRPC_PORT
        self.QDRANT_TIMEOUT = QDRANT_TIMEOUT
        self.QDRANT_HNSW_M = QDRANT_HNSW_M

        if not self.QDRANT_URI:
            raise ValueError('QDRANT_URI is not set. Please configure it in the environment variables.')

        # Unified handling for either scheme
        parsed = urlparse(self.QDRANT_URI)
        host = parsed.hostname or self.QDRANT_URI
        http_port = parsed.port or 6333  # default REST port

        self.client = (
            Qclient(
                host=host,
                port=http_port,
                grpc_port=self.GRPC_PORT,
                prefer_grpc=self.PREFER_GRPC,
                api_key=self.QDRANT_API_KEY,
                timeout=self.QDRANT_TIMEOUT,
            )
            if self.PREFER_GRPC
            else Qclient(
                url=self.QDRANT_URI,
                api_key=self.QDRANT_API_KEY,
                timeout=self.QDRANT_TIMEOUT,
            )
        )

        # Shared multi-tenant collections store each embedding size as a named
        # vector whose name is the dimension (for example "1536"). Legacy
        # collections with an unnamed default vector are still accepted when
        # that size matches. Requires Qdrant >= 1.18 to add new sizes.
        self.MEMORY_COLLECTION = f'{self.collection_prefix}_memories'
        self.KNOWLEDGE_COLLECTION = f'{self.collection_prefix}_knowledge'
        self.FILE_COLLECTION = f'{self.collection_prefix}_files'
        self.WEB_SEARCH_COLLECTION = f'{self.collection_prefix}_web-search'
        self.HASH_BASED_COLLECTION = f'{self.collection_prefix}_hash-based'

    def _result_to_get_result(self, points) -> GetResult:
        ids, documents, metadatas = [], [], []
        for point in points:
            payload = point.payload
            ids.append(point.id)
            documents.append(payload['text'])
            metadatas.append(payload['metadata'])
        return GetResult(ids=[ids], documents=[documents], metadatas=[metadatas])

    def _scroll_points(self, collection_name: str, scroll_filter: models.Filter, limit: Optional[int] = None) -> List:
        # Paged so a strict-mode max_query_limit does not reject the read
        points, offset = [], None
        while True:
            page_size = SCROLL_PAGE_SIZE if limit is None else min(SCROLL_PAGE_SIZE, limit - len(points))
            page, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=page_size,
                offset=offset,
            )
            points.extend(page)
            if offset is None or len(points) == limit:
                return points

    def _get_collection_and_tenant_id(self, collection_name: str) -> Tuple[str, str]:
        """
        Maps the traditional collection name to multi-tenant collection and tenant ID.

        Returns:
            tuple: (collection_name, tenant_id)

        WARNING: This mapping relies on current Open WebUI naming conventions for
        collection names. If Open WebUI changes how it generates collection names
        (e.g., "user-memory-" prefix, "file-" prefix, web search patterns, or hash
        formats), this mapping will break and route data to incorrect collections.
        POTENTIALLY CAUSING HUGE DATA CORRUPTION, DATA CONSISTENCY ISSUES AND INCORRECT
        DATA MAPPING INSIDE THE DATABASE.
        """
        # Check for user memory collections
        tenant_id = collection_name

        if collection_name.startswith('user-memory-'):
            return self.MEMORY_COLLECTION, tenant_id

        # Check for file collections
        elif collection_name.startswith('file-'):
            return self.FILE_COLLECTION, tenant_id

        # Check for web search collections
        elif collection_name.startswith('web-search-'):
            return self.WEB_SEARCH_COLLECTION, tenant_id

        # Handle hash-based collections (YouTube and web URLs)
        elif len(collection_name) == 63 and all(c in '0123456789abcdef' for c in collection_name):
            return self.HASH_BASED_COLLECTION, tenant_id

        else:
            return self.KNOWLEDGE_COLLECTION, tenant_id

    def _create_multi_tenant_collection(self, mt_collection_name: str, dimension: int):
        """
        Creates a collection with multi-tenancy configuration and payload indexes for tenant_id and metadata fields.
        """
        vector_name = str(dimension)
        self.client.create_collection(
            collection_name=mt_collection_name,
            vectors_config={
                vector_name: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=self.QDRANT_ON_DISK,
                )
            },
            # Disable global index building due to multitenancy
            # For more details https://qdrant.tech/documentation/guides/multiple-partitions/#calibrate-performance
            hnsw_config=models.HnswConfigDiff(
                payload_m=self.QDRANT_HNSW_M,
                m=0,
            ),
        )
        log.info(
            'Multi-tenant collection %s created with named vector %s!',
            mt_collection_name,
            vector_name,
        )

        self.client.create_payload_index(
            collection_name=mt_collection_name,
            field_name=TENANT_ID_FIELD,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
                on_disk=self.QDRANT_ON_DISK,
            ),
        )

        for field in ('metadata.hash', 'metadata.file_id'):
            self.client.create_payload_index(
                collection_name=mt_collection_name,
                field_name=field,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    on_disk=self.QDRANT_ON_DISK,
                ),
            )

    def _vector_schemas(self, collection_name: str) -> Dict[int, Optional[str]]:
        """Map dimension -> vector column name. `None` is the unnamed default."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
        except Exception:
            log.debug('Could not read vector schema for %s', collection_name, exc_info=True)
            return {}
        params = getattr(getattr(info, 'config', None), 'params', None)
        vectors = getattr(params, 'vectors', None)
        # Missing collection params or a collection with no dense vectors configured.
        if vectors is None:
            return {}
        # Legacy single unnamed vector: Qdrant returns VectorParams, not a name->params dict.
        if not isinstance(vectors, dict):
            size = getattr(vectors, 'size', None)
            return {size: None} if isinstance(size, int) else {}
        schemas: Dict[int, Optional[str]] = {}
        for name, vector_params in vectors.items():
            size = getattr(vector_params, 'size', None)
            # Skip entries that are not dense vectors with a known size (e.g. sparse-only).
            if not isinstance(size, int):
                continue
            # Unnamed default uses "" (or None); named columns always have a different size.
            if not name:
                schemas[size] = None
            else:
                schemas[size] = str(name)
        return schemas

    def _create_named_vector(self, collection_name: str, vector_name: str, dimension: int) -> None:
        """Create a named vector column on the collection."""
        self.client.create_vector_name(
            collection_name=collection_name,
            vector_name=vector_name,
            vector_name_config=models.DenseVectorNameConfig(
                dense=models.DenseVectorConfig(
                    size=dimension,
                    distance=models.Distance.COSINE,
                )
            ),
        )
        log.info('Added named vector %s to Qdrant collection %s', vector_name, collection_name)

    def _ensure_vector_column(self, collection_name: str, dimension: int) -> Optional[str]:
        """Return the vector column for `dimension`, creating a named one if needed."""
        schemas = self._vector_schemas(collection_name)
        if dimension in schemas:
            return schemas[dimension]
        vector_name = str(dimension)
        self._create_named_vector(collection_name, vector_name, dimension)
        return vector_name

    def _create_points(
        self, items: List[VectorItem], tenant_id: str, vector_name: Optional[str]
    ) -> List[PointStruct]:
        """
        Create point structs from vector items with tenant ID.

        `vector_name` is `None` for the unnamed default, or a named vector such as `"1536"`.
        """
        return [
            PointStruct(
                id=item['id'],
                vector={vector_name: item['vector']} if vector_name else item['vector'],
                payload={
                    'text': item['text'],
                    'metadata': process_metadata(item['metadata']),
                    TENANT_ID_FIELD: tenant_id,
                },
            )
            for item in items
        ]

    def _ensure_collection(self, mt_collection_name: str, dimension: int):
        """
        Ensure the collection exists and payload indexes are created for tenant_id and metadata fields.
        """
        if not self.client.collection_exists(collection_name=mt_collection_name):
            self._create_multi_tenant_collection(mt_collection_name, dimension)

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a logical collection exists by checking for any points with the tenant ID.
        """
        if not self.client:
            return False
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            return False
        tenant_filter = _tenant_filter(tenant_id)
        count_result = self.client.count(
            collection_name=mt_collection,
            count_filter=models.Filter(must=[tenant_filter]),
        )
        return count_result.count > 0

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Delete vectors by ID or filter from a collection with tenant isolation.
        """
        if not self.client:
            return None

        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, nothing to delete", mt_collection)
            return None

        must_conditions = [_tenant_filter(tenant_id)]
        if ids:
            # Delete by point ID within the tenant. The point ID is the item's id
            # (see _create_points); filtering on metadata.id silently misses points
            # whose payload omits an id (e.g. memories), leaving orphaned vectors.
            must_conditions.append(models.HasIdCondition(has_id=ids))
        elif filter:
            must_conditions += [_metadata_filter(k, '$eq', v) for k, v in filter.items()]

        return self.client.delete(
            collection_name=mt_collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=must_conditions)),
        )

    def search(
        self,
        collection_name: str,
        vectors: List[List[float | int]],
        filter: Optional[Dict] = None,
        limit: int = 10,
    ) -> Optional[SearchResult]:
        """
        Search for the nearest neighbor items based on the vectors with tenant isolation.
        """
        if not self.client or not vectors:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, search returns None", mt_collection)
            return None
        dimension = len(vectors[0])
        schemas = self._vector_schemas(mt_collection)
        if dimension not in schemas:
            log.debug(
                'Collection %s has no vector slot for dimension %s, search returns None',
                mt_collection,
                dimension,
            )
            return None
        vector_name = schemas[dimension]

        conditions = [_tenant_filter(tenant_id)]
        if filter:
            conditions.extend(_metadata_filter(key, op, value) for key, op, value in iter_filter_conditions(filter))
        query_response = self.client.query_points(
            collection_name=mt_collection,
            query=vectors[0],
            using=vector_name,
            limit=limit,
            query_filter=models.Filter(must=conditions),
        )
        get_result = self._result_to_get_result(query_response.points)
        return SearchResult(
            ids=get_result.ids,
            documents=get_result.documents,
            metadatas=get_result.metadatas,
            distances=[[(point.score + 1.0) / 2.0 for point in query_response.points]],
        )

    def query(self, collection_name: str, filter: Dict[str, Any], limit: Optional[int] = None):
        """
        Query points with filters and tenant isolation.
        """
        if not self.client:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, query returns None", mt_collection)
            return None
        tenant_filter = _tenant_filter(tenant_id)
        field_conditions = [_metadata_filter(k, '$eq', v) for k, v in filter.items()]
        combined_filter = models.Filter(must=[tenant_filter, *field_conditions])
        points = self._scroll_points(mt_collection, combined_filter, limit)
        return self._result_to_get_result(points)

    def get(self, collection_name: str) -> Optional[GetResult]:
        """
        Get all items in a collection with tenant isolation.
        """
        if not self.client:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, get returns None", mt_collection)
            return None
        tenant_filter = _tenant_filter(tenant_id)
        points = self._scroll_points(mt_collection, models.Filter(must=[tenant_filter]))
        return self._result_to_get_result(points)

    def upsert(self, collection_name: str, items: List[VectorItem]):
        """
        Upsert items with tenant ID.
        """
        if not self.client or not items:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        dimension = len(items[0]['vector'])
        self._ensure_collection(mt_collection, dimension)
        vector_name = self._ensure_vector_column(mt_collection, dimension)
        points = self._create_points(items, tenant_id, vector_name)
        self.client.upload_points(mt_collection, points)
        return None

    def insert(self, collection_name: str, items: List[VectorItem]):
        """
        Insert items with tenant ID.
        """
        return self.upsert(collection_name, items)

    def reset(self):
        """
        Reset the database by deleting all collections.
        """
        if not self.client:
            return None
        for collection in self.client.get_collections().collections:
            if collection.name.startswith(self.collection_prefix):
                self.client.delete_collection(collection_name=collection.name)

    def delete_collection(self, collection_name: str):
        """
        Delete a collection.
        """
        if not self.client:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, nothing to delete", mt_collection)
            return None
        self.client.delete(
            collection_name=mt_collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[_tenant_filter(tenant_id)])),
        )
