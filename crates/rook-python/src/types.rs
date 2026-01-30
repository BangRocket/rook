//! Python-compatible types that mirror rook_core types.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// A memory item stored in the system.
#[pyclass]
#[derive(Clone)]
pub struct MemoryItem {
    /// Unique identifier for the memory.
    #[pyo3(get)]
    pub id: String,
    /// The memory content/text.
    #[pyo3(get)]
    pub memory: String,
    /// MD5 hash of the memory content.
    #[pyo3(get)]
    pub hash: Option<String>,
    /// Similarity score (from search).
    #[pyo3(get)]
    pub score: Option<f32>,
    /// Custom metadata as JSON string (use .metadata_dict() to get as dict).
    metadata_json: String,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: Option<String>,
    /// Last update timestamp.
    #[pyo3(get)]
    pub updated_at: Option<String>,
    /// Category assigned to this memory.
    #[pyo3(get)]
    pub category: Option<String>,
    /// Whether this is a key/important memory.
    #[pyo3(get)]
    pub is_key: bool,
}

#[pymethods]
impl MemoryItem {
    /// Get metadata as a Python dict.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&self.metadata_json) {
            if let serde_json::Value::Object(map) = value {
                for (k, v) in map {
                    if let Ok(val) = json_to_python(py, &v) {
                        dict.set_item(k, val)?;
                    }
                }
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryItem(id='{}', memory='{}', score={:?})",
            self.id,
            if self.memory.len() > 50 {
                format!("{}...", &self.memory[..50])
            } else {
                self.memory.clone()
            },
            self.score
        )
    }
}

impl MemoryItem {
    /// Create a new MemoryItem from rook_core::MemoryItem.
    pub fn from_core(item: rook_core::MemoryItem) -> Self {
        let metadata_json = item
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());

        Self {
            id: item.id,
            memory: item.memory,
            hash: item.hash,
            score: item.score,
            metadata_json,
            created_at: item.created_at,
            updated_at: item.updated_at,
            category: item.category,
            is_key: item.is_key,
        }
    }
}

/// Result of a search operation.
#[pyclass]
#[derive(Clone)]
pub struct SearchResult {
    /// Unique identifier for the memory.
    #[pyo3(get)]
    pub id: String,
    /// The memory content/text.
    #[pyo3(get)]
    pub memory: String,
    /// Similarity score.
    #[pyo3(get)]
    pub score: f32,
    /// Custom metadata as JSON string.
    metadata_json: String,
}

#[pymethods]
impl SearchResult {
    /// Get metadata as a Python dict.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&self.metadata_json) {
            if let serde_json::Value::Object(map) = value {
                for (k, v) in map {
                    if let Ok(val) = json_to_python(py, &v) {
                        dict.set_item(k, val)?;
                    }
                }
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', score={:.4}, memory='{}')",
            self.id,
            self.score,
            if self.memory.len() > 50 {
                format!("{}...", &self.memory[..50])
            } else {
                self.memory.clone()
            }
        )
    }
}

impl SearchResult {
    /// Create a SearchResult from a MemoryItem.
    pub fn from_memory_item(item: &rook_core::MemoryItem) -> Self {
        let metadata_json = item
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());

        Self {
            id: item.id.clone(),
            memory: item.memory.clone(),
            score: item.score.unwrap_or(0.0),
            metadata_json,
        }
    }
}

/// Result of an add operation.
#[pyclass]
#[derive(Clone)]
pub struct AddResult {
    /// Memory items that were created/updated.
    #[pyo3(get)]
    pub memories: Vec<MemoryItem>,
}

#[pymethods]
impl AddResult {
    fn __repr__(&self) -> String {
        format!("AddResult(count={})", self.memories.len())
    }

    fn __len__(&self) -> usize {
        self.memories.len()
    }
}

impl AddResult {
    /// Create an AddResult from rook_core::AddResult.
    pub fn from_core(result: &rook_core::AddResult) -> Self {
        let memories: Vec<MemoryItem> = result
            .results
            .iter()
            .map(|r| {
                let item = rook_core::MemoryItem::new(r.id.clone(), r.memory.clone());
                MemoryItem::from_core(item)
            })
            .collect();

        Self { memories }
    }
}

/// Convert serde_json::Value to a Python object.
pub fn json_to_python(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    use pyo3::conversion::IntoPyObject;
    use pyo3::types::{PyBool, PyString};

    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => {
            let py_bool = PyBool::new(py, *b);
            Ok(py_bool.to_owned().unbind().into_any())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                // Use IntoPyObject for integers
                let py_int = i.into_pyobject(py)?.to_owned();
                Ok(py_int.unbind().into_any())
            } else if let Some(f) = n.as_f64() {
                let py_float = f.into_pyobject(py)?.to_owned();
                Ok(py_float.unbind().into_any())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => {
            Ok(PyString::new(py, s).unbind().into_any())
        }
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr
                .iter()
                .filter_map(|v| json_to_python(py, v).ok())
                .collect();
            let list = pyo3::types::PyList::new(py, &items)?;
            Ok(list.unbind().into_any())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                if let Ok(val) = json_to_python(py, v) {
                    dict.set_item(k, val)?;
                }
            }
            Ok(dict.unbind().into_any())
        }
    }
}

/// Convert a Python object to serde_json::Value.
pub fn python_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::json!(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let arr: Vec<serde_json::Value> = list
            .iter()
            .filter_map(|item| python_to_json(py, &item).ok())
            .collect();
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict {
            if let Ok(key) = k.extract::<String>() {
                if let Ok(val) = python_to_json(py, &v) {
                    map.insert(key, val);
                }
            }
        }
        Ok(serde_json::Value::Object(map))
    } else {
        // Fallback: convert to string representation
        let s = obj.str()?.to_string();
        Ok(serde_json::Value::String(s))
    }
}

/// Convert a Python dict to a HashMap<String, serde_json::Value>.
pub fn python_dict_to_metadata(
    py: Python<'_>,
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<HashMap<String, serde_json::Value>>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            if let Ok(dict) = o.downcast::<PyDict>() {
                let mut map = HashMap::new();
                for (k, v) in dict {
                    if let Ok(key) = k.extract::<String>() {
                        if let Ok(val) = python_to_json(py, &v) {
                            map.insert(key, val);
                        }
                    }
                }
                Ok(Some(map))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "metadata must be a dict",
                ))
            }
        }
    }
}
