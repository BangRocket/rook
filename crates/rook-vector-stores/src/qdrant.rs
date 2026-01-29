//! Qdrant vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{
    CollectionInfo, DistanceMetric, VectorRecord, VectorSearchResult, VectorStore,
    VectorStoreConfig,
};
use rook_core::types::Filter;

use qdrant_client::qdrant::{
    vectors_config::Config, Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance,
    FieldCondition, Filter as QdrantFilter, GetPointsBuilder, Match, PointId, PointStruct,
    Range, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder, Value,
    VectorParamsBuilder,
};
use qdrant_client::Qdrant;

/// Qdrant vector store implementation.
pub struct QdrantVectorStore {
    client: Qdrant,
    config: VectorStoreConfig,
}

impl QdrantVectorStore {
    /// Create a new Qdrant vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .config
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("http://localhost:6334");

        let api_key = config.config.get("api_key").and_then(|v| v.as_str());

        let client = if let Some(key) = api_key {
            Qdrant::from_url(url)
                .api_key(key)
                .build()
                .map_err(|e| RookError::vector_store(format!("Failed to create Qdrant client: {}", e)))?
        } else {
            Qdrant::from_url(url)
                .build()
                .map_err(|e| RookError::vector_store(format!("Failed to create Qdrant client: {}", e)))?
        };

        Ok(Self { client, config })
    }

    fn distance_to_qdrant(metric: DistanceMetric) -> Distance {
        match metric {
            DistanceMetric::Cosine => Distance::Cosine,
            DistanceMetric::Euclidean => Distance::Euclid,
            DistanceMetric::DotProduct => Distance::Dot,
            DistanceMetric::Manhattan => Distance::Manhattan,
        }
    }

    fn qdrant_to_distance(distance: i32) -> DistanceMetric {
        match Distance::try_from(distance) {
            Ok(Distance::Cosine) => DistanceMetric::Cosine,
            Ok(Distance::Euclid) => DistanceMetric::Euclidean,
            Ok(Distance::Dot) => DistanceMetric::DotProduct,
            Ok(Distance::Manhattan) => DistanceMetric::Manhattan,
            _ => DistanceMetric::Cosine,
        }
    }

    fn payload_to_hashmap(payload: HashMap<String, Value>) -> HashMap<String, serde_json::Value> {
        payload
            .into_iter()
            .map(|(k, v)| (k, Self::qdrant_value_to_json(v)))
            .collect()
    }

    fn qdrant_value_to_json(value: Value) -> serde_json::Value {
        use qdrant_client::qdrant::value::Kind;
        match value.kind {
            Some(Kind::NullValue(_)) => serde_json::Value::Null,
            Some(Kind::BoolValue(b)) => serde_json::Value::Bool(b),
            Some(Kind::IntegerValue(i)) => serde_json::Value::Number(i.into()),
            Some(Kind::DoubleValue(d)) => serde_json::Number::from_f64(d)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Some(Kind::StringValue(s)) => serde_json::Value::String(s),
            Some(Kind::ListValue(list)) => serde_json::Value::Array(
                list.values
                    .into_iter()
                    .map(Self::qdrant_value_to_json)
                    .collect(),
            ),
            Some(Kind::StructValue(s)) => serde_json::Value::Object(
                s.fields
                    .into_iter()
                    .map(|(k, v)| (k, Self::qdrant_value_to_json(v)))
                    .collect(),
            ),
            None => serde_json::Value::Null,
        }
    }

    fn json_to_qdrant_value(value: serde_json::Value) -> Value {
        use qdrant_client::qdrant::value::Kind;
        use qdrant_client::qdrant::{ListValue, Struct};

        let kind = match value {
            serde_json::Value::Null => Kind::NullValue(0),
            serde_json::Value::Bool(b) => Kind::BoolValue(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Kind::IntegerValue(i)
                } else if let Some(f) = n.as_f64() {
                    Kind::DoubleValue(f)
                } else {
                    Kind::NullValue(0)
                }
            }
            serde_json::Value::String(s) => Kind::StringValue(s),
            serde_json::Value::Array(arr) => Kind::ListValue(ListValue {
                values: arr.into_iter().map(Self::json_to_qdrant_value).collect(),
            }),
            serde_json::Value::Object(obj) => Kind::StructValue(Struct {
                fields: obj
                    .into_iter()
                    .map(|(k, v)| (k, Self::json_to_qdrant_value(v)))
                    .collect(),
            }),
        };

        Value { kind: Some(kind) }
    }

    fn convert_filter(filter: &Filter) -> QdrantFilter {
        match filter {
            Filter::Condition(cond) => {
                let condition = Self::convert_condition(cond);
                QdrantFilter {
                    must: vec![condition],
                    ..Default::default()
                }
            }
            Filter::And(filters) => {
                let must: Vec<Condition> = filters
                    .iter()
                    .flat_map(|f| {
                        let qf = Self::convert_filter(f);
                        qf.must.into_iter()
                    })
                    .collect();
                QdrantFilter {
                    must,
                    ..Default::default()
                }
            }
            Filter::Or(filters) => {
                let should: Vec<Condition> = filters
                    .iter()
                    .map(|f| {
                        let qf = Self::convert_filter(f);
                        Condition {
                            condition_one_of: Some(
                                qdrant_client::qdrant::condition::ConditionOneOf::Filter(qf),
                            ),
                        }
                    })
                    .collect();
                QdrantFilter {
                    should,
                    ..Default::default()
                }
            }
            Filter::Not(inner) => {
                let inner_filter = Self::convert_filter(inner);
                QdrantFilter {
                    must_not: inner_filter.must,
                    ..Default::default()
                }
            }
        }
    }

    fn convert_condition(cond: &rook_core::types::FilterCondition) -> Condition {
        use rook_core::types::FilterOperator;

        let field_condition = match &cond.operator {
            FilterOperator::Eq(value) => {
                let match_value = Self::value_to_match(value);
                FieldCondition {
                    key: cond.field.clone(),
                    r#match: Some(match_value),
                    ..Default::default()
                }
            }
            FilterOperator::Gt(value) => FieldCondition {
                key: cond.field.clone(),
                range: Some(Range {
                    gt: value.as_f64(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            FilterOperator::Gte(value) => FieldCondition {
                key: cond.field.clone(),
                range: Some(Range {
                    gte: value.as_f64(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            FilterOperator::Lt(value) => FieldCondition {
                key: cond.field.clone(),
                range: Some(Range {
                    lt: value.as_f64(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            FilterOperator::Lte(value) => FieldCondition {
                key: cond.field.clone(),
                range: Some(Range {
                    lte: value.as_f64(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            FilterOperator::Ne(value) => {
                // Ne is typically handled with must_not, but we approximate
                let match_value = Self::value_to_match(value);
                FieldCondition {
                    key: cond.field.clone(),
                    r#match: Some(match_value),
                    ..Default::default()
                }
            }
            FilterOperator::In(values) => {
                // Match any of the values
                let keywords: Vec<String> = values
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                FieldCondition {
                    key: cond.field.clone(),
                    r#match: Some(Match {
                        match_value: Some(
                            qdrant_client::qdrant::r#match::MatchValue::Keywords(
                                qdrant_client::qdrant::RepeatedStrings { strings: keywords },
                            ),
                        ),
                    }),
                    ..Default::default()
                }
            }
            FilterOperator::Contains(text) => FieldCondition {
                key: cond.field.clone(),
                r#match: Some(Match {
                    match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Text(
                        text.clone(),
                    )),
                }),
                ..Default::default()
            },
            _ => FieldCondition {
                key: cond.field.clone(),
                ..Default::default()
            },
        };

        Condition {
            condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                field_condition,
            )),
        }
    }

    fn value_to_match(value: &serde_json::Value) -> Match {
        match value {
            serde_json::Value::String(s) => Match {
                match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                    s.clone(),
                )),
            },
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Match {
                        match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Integer(i)),
                    }
                } else {
                    Match { match_value: None }
                }
            }
            serde_json::Value::Bool(b) => Match {
                match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Boolean(*b)),
            },
            _ => Match { match_value: None },
        }
    }

    fn extract_point_id(point_id: Option<PointId>) -> String {
        match point_id {
            Some(PointId {
                point_id_options:
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)),
            }) => uuid,
            Some(PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)),
            }) => num.to_string(),
            _ => String::new(),
        }
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance: DistanceMetric,
    ) -> RookResult<()> {
        let request = CreateCollectionBuilder::new(name).vectors_config(
            VectorParamsBuilder::new(dimension as u64, Self::distance_to_qdrant(distance)),
        );

        self.client
            .create_collection(request)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create collection: {}", e)))?;

        Ok(())
    }

    async fn insert(&self, records: Vec<VectorRecord>) -> RookResult<()> {
        let points: Vec<PointStruct> = records
            .into_iter()
            .map(|record| {
                let payload: HashMap<String, Value> = record
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, Self::json_to_qdrant_value(v)))
                    .collect();

                PointStruct::new(record.id, record.vector, payload)
            })
            .collect();

        let request = UpsertPointsBuilder::new(self.collection_name(), points);

        self.client
            .upsert_points(request)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to insert vectors: {}", e)))?;

        Ok(())
    }

    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        filters: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        let mut request = SearchPointsBuilder::new(
            self.collection_name(),
            query_vector.to_vec(),
            limit as u64,
        )
        .with_payload(true);

        if let Some(f) = filters {
            request = request.filter(Self::convert_filter(&f));
        }

        let search_result = self
            .client
            .search_points(request)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search vectors: {}", e)))?;

        let results = search_result
            .result
            .into_iter()
            .map(|point| VectorSearchResult {
                id: Self::extract_point_id(point.id),
                score: point.score,
                payload: Self::payload_to_hashmap(point.payload),
            })
            .collect();

        Ok(results)
    }

    async fn get(&self, id: &str) -> RookResult<Option<VectorRecord>> {
        let request = GetPointsBuilder::new(self.collection_name(), vec![id.into()])
            .with_payload(true)
            .with_vectors(true);

        let result = self
            .client
            .get_points(request)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get vector: {}", e)))?;

        let record = result.result.into_iter().next().map(|point| {
            let vector = match point.vectors {
                Some(vectors) => match vectors.vectors_options {
                    Some(
                        qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(v),
                    ) => v.data,
                    _ => vec![],
                },
                None => vec![],
            };

            VectorRecord {
                id: id.to_string(),
                vector,
                payload: Self::payload_to_hashmap(point.payload),
                score: None,
            }
        });

        Ok(record)
    }

    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<()> {
        // Get existing record
        let existing = self.get(id).await?;
        let existing = existing.ok_or_else(|| RookError::not_found(id))?;

        // Merge updates
        let new_vector = vector.unwrap_or(existing.vector);
        let new_payload = if let Some(p) = payload {
            let mut merged = existing.payload;
            merged.extend(p);
            merged
        } else {
            existing.payload
        };

        // Upsert with updated values
        let record = VectorRecord {
            id: id.to_string(),
            vector: new_vector,
            payload: new_payload,
            score: None,
        };

        self.insert(vec![record]).await
    }

    async fn delete(&self, id: &str) -> RookResult<()> {
        let point_id: PointId = id.into();
        let request = DeletePointsBuilder::new(self.collection_name())
            .points(vec![point_id]);

        self.client
            .delete_points(request)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete vector: {}", e)))?;

        Ok(())
    }

    async fn list(
        &self,
        filters: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        let mut request = ScrollPointsBuilder::new(self.collection_name())
            .with_payload(true)
            .with_vectors(true);

        if let Some(f) = filters {
            request = request.filter(Self::convert_filter(&f));
        }
        if let Some(l) = limit {
            request = request.limit(l as u32);
        }

        let scroll_result = self
            .client
            .scroll(request)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list vectors: {}", e)))?;

        let records = scroll_result
            .result
            .into_iter()
            .map(|point| {
                let id = Self::extract_point_id(point.id);
                let vector = match point.vectors {
                    Some(vectors) => match vectors.vectors_options {
                        Some(
                            qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(v),
                        ) => v.data,
                        _ => vec![],
                    },
                    None => vec![],
                };

                VectorRecord {
                    id,
                    vector,
                    payload: Self::payload_to_hashmap(point.payload),
                    score: None,
                }
            })
            .collect();

        Ok(records)
    }

    async fn list_collections(&self) -> RookResult<Vec<String>> {
        let collections = self
            .client
            .list_collections()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list collections: {}", e)))?;

        Ok(collections
            .collections
            .into_iter()
            .map(|c| c.name)
            .collect())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        self.client
            .delete_collection(name)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete collection: {}", e)))?;

        Ok(())
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let response = self
            .client
            .collection_info(name)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get collection info: {}", e)))?;

        let info = response.result.ok_or_else(|| {
            RookError::vector_store("No collection info in response".to_string())
        })?;

        let (dimension, distance) = match &info.config {
            Some(config) => match &config.params {
                Some(params) => match &params.vectors_config {
                    Some(vc) => match &vc.config {
                        Some(Config::Params(p)) => (
                            p.size as usize,
                            Self::qdrant_to_distance(p.distance),
                        ),
                        _ => (0, DistanceMetric::Cosine),
                    },
                    None => (0, DistanceMetric::Cosine),
                },
                None => (0, DistanceMetric::Cosine),
            },
            None => (0, DistanceMetric::Cosine),
        };

        Ok(CollectionInfo {
            name: name.to_string(),
            vector_count: info.points_count.unwrap_or(0),
            dimension,
            distance,
        })
    }

    async fn reset(&self) -> RookResult<()> {
        let name = self.collection_name();
        let info = self.collection_info(name).await?;

        self.delete_collection(name).await?;
        self.create_collection(name, info.dimension, info.distance)
            .await?;

        Ok(())
    }

    fn collection_name(&self) -> &str {
        &self.config.collection_name
    }
}
