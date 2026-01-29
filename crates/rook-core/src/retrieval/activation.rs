//! Spreading activation algorithm for memory graph traversal.
//!
//! Based on Collins & Loftus (1975) spreading activation theory,
//! adapted from SA-RAG (arXiv:2512.15922) with bounded propagation.
//!
//! # Example with EmbeddedGraphStore
//!
//! ```ignore
//! use rook_core::retrieval::{spread_activation_by_id, SpreadingConfig};
//!
//! let store = EmbeddedGraphStore::new("graph.db")?;
//! let graph = store.graph.read().unwrap();
//! let index = store.node_index.read().unwrap();
//!
//! let activated = spread_activation_by_id(
//!     &graph,
//!     &index,
//!     &[("seed-memory-id".to_string(), 1.0)],
//!     &SpreadingConfig::default(),
//!     |node| node.id.to_string(),
//!     |edge| edge.weight as f32,
//! );
//! ```

use std::collections::{HashMap, VecDeque};

use ordered_float::OrderedFloat;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};

use super::config::SpreadingConfig;

/// A memory that has been activated through spreading activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedMemory {
    /// Memory/entity ID that was activated.
    pub memory_id: String,
    /// Activation level (0.0-1.0).
    pub activation: f32,
    /// Depth at which this node was reached.
    pub depth: usize,
}

/// Node data required for spreading activation.
pub trait ActivationNode {
    fn id(&self) -> &str;
}

/// Edge data required for spreading activation.
pub trait ActivationEdge {
    fn weight(&self) -> f32;
}

/// Spread activation from seed nodes through the graph.
///
/// # Arguments
/// * `graph` - The graph to traverse (petgraph DiGraph)
/// * `node_index` - Map from node ID to NodeIndex for fast lookup
/// * `seeds` - Initial (node_id, activation) pairs to start propagation
/// * `config` - Spreading activation configuration
///
/// # Returns
/// Vector of activated memories sorted by activation (descending).
///
/// # Algorithm
/// BFS propagation from seeds with:
/// - Activation decays by decay_factor per hop
/// - Edge weights modulate propagation
/// - High-degree nodes receive fan-out penalty
/// - Stops when activation < firing_threshold or depth > max_depth
pub fn spread_activation<N, E>(
    graph: &DiGraph<N, E>,
    node_index: &HashMap<String, NodeIndex>,
    seeds: &[(String, f32)],
    config: &SpreadingConfig,
) -> Vec<ActivatedMemory>
where
    N: ActivationNode,
    E: ActivationEdge,
{
    let mut activation: HashMap<NodeIndex, f32> = HashMap::new();
    let mut visited_depth: HashMap<NodeIndex, usize> = HashMap::new();
    let mut queue: VecDeque<(NodeIndex, usize, f32)> = VecDeque::new();

    // Initialize seeds
    for (id, initial_act) in seeds {
        if let Some(&idx) = node_index.get(id) {
            let clamped_act = initial_act.clamp(0.0, 1.0);
            activation.insert(idx, clamped_act);
            visited_depth.insert(idx, 0);
            queue.push_back((idx, 0, clamped_act));
        }
    }

    // BFS propagation with decay
    while let Some((current, depth, current_act)) = queue.pop_front() {
        // Stop at max depth
        if depth >= config.max_depth {
            continue;
        }

        // Get neighbors and calculate fan-out penalty
        let neighbors: Vec<NodeIndex> = graph.neighbors(current).collect();
        let degree = neighbors.len() as f32;
        let fan_out_factor = 1.0 / (1.0 + config.fan_out_penalty * degree);

        for neighbor in neighbors {
            // Get edge weight
            let edge_weight = graph
                .find_edge(current, neighbor)
                .map(|e| graph[e].weight())
                .unwrap_or(1.0);

            // Calculate propagated activation
            let propagated = current_act * edge_weight * config.decay_factor * fan_out_factor;

            // Skip if below threshold
            if propagated < config.firing_threshold {
                continue;
            }

            let new_depth = depth + 1;
            let entry = activation.entry(neighbor).or_insert(0.0);

            // Accumulate activation (cap at 1.0)
            if propagated > *entry {
                *entry = (*entry + propagated).min(1.0);

                // Queue if this is a better path
                if visited_depth.get(&neighbor).map_or(true, |&d| new_depth < d) {
                    visited_depth.insert(neighbor, new_depth);
                    queue.push_back((neighbor, new_depth, *entry));
                }
            }
        }
    }

    // Convert to results, sorted by activation descending
    let mut results: Vec<ActivatedMemory> = activation
        .into_iter()
        .filter(|(_, act)| *act >= config.firing_threshold)
        .filter_map(|(idx, act)| {
            let depth = visited_depth.get(&idx).copied().unwrap_or(0);
            Some(ActivatedMemory {
                memory_id: graph[idx].id().to_string(),
                activation: act,
                depth,
            })
        })
        .collect();

    // Sort by activation descending (OrderedFloat handles NaN)
    results.sort_by(|a, b| OrderedFloat(b.activation).cmp(&OrderedFloat(a.activation)));

    results
}

/// Convenience function for graphs where nodes have a `memory_id: String` field.
/// This avoids needing to implement traits on external types.
///
/// # Arguments
/// * `graph` - The graph to traverse (petgraph DiGraph)
/// * `node_index` - Map from node ID to NodeIndex for fast lookup
/// * `seeds` - Initial (node_id, activation) pairs to start propagation
/// * `config` - Spreading activation configuration
/// * `get_id` - Closure to extract ID from node
/// * `get_weight` - Closure to extract weight from edge
///
/// # Returns
/// Vector of activated memories sorted by activation (descending).
pub fn spread_activation_by_id<N, E, F, G>(
    graph: &DiGraph<N, E>,
    node_index: &HashMap<String, NodeIndex>,
    seeds: &[(String, f32)],
    config: &SpreadingConfig,
    get_id: F,
    get_weight: G,
) -> Vec<ActivatedMemory>
where
    F: Fn(&N) -> String,
    G: Fn(&E) -> f32,
{
    let mut activation: HashMap<NodeIndex, f32> = HashMap::new();
    let mut visited_depth: HashMap<NodeIndex, usize> = HashMap::new();
    let mut queue: VecDeque<(NodeIndex, usize, f32)> = VecDeque::new();

    // Initialize seeds
    for (id, initial_act) in seeds {
        if let Some(&idx) = node_index.get(id) {
            let clamped_act = initial_act.clamp(0.0, 1.0);
            activation.insert(idx, clamped_act);
            visited_depth.insert(idx, 0);
            queue.push_back((idx, 0, clamped_act));
        }
    }

    // BFS propagation with decay
    while let Some((current, depth, current_act)) = queue.pop_front() {
        if depth >= config.max_depth {
            continue;
        }

        let neighbors: Vec<NodeIndex> = graph.neighbors(current).collect();
        let degree = neighbors.len() as f32;
        let fan_out_factor = 1.0 / (1.0 + config.fan_out_penalty * degree);

        for neighbor in neighbors {
            let edge_weight = graph
                .find_edge(current, neighbor)
                .map(|e| get_weight(&graph[e]))
                .unwrap_or(1.0);

            let propagated = current_act * edge_weight * config.decay_factor * fan_out_factor;

            if propagated < config.firing_threshold {
                continue;
            }

            let new_depth = depth + 1;
            let entry = activation.entry(neighbor).or_insert(0.0);

            if propagated > *entry {
                *entry = (*entry + propagated).min(1.0);

                if visited_depth.get(&neighbor).map_or(true, |&d| new_depth < d) {
                    visited_depth.insert(neighbor, new_depth);
                    queue.push_back((neighbor, new_depth, *entry));
                }
            }
        }
    }

    let mut results: Vec<ActivatedMemory> = activation
        .into_iter()
        .filter(|(_, act)| *act >= config.firing_threshold)
        .filter_map(|(idx, act)| {
            let depth = visited_depth.get(&idx).copied().unwrap_or(0);
            Some(ActivatedMemory {
                memory_id: get_id(&graph[idx]),
                activation: act,
                depth,
            })
        })
        .collect();

    results.sort_by(|a, b| OrderedFloat(b.activation).cmp(&OrderedFloat(a.activation)));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test node
    #[derive(Debug, Clone)]
    struct TestNode {
        id: String,
    }

    impl ActivationNode for TestNode {
        fn id(&self) -> &str {
            &self.id
        }
    }

    // Simple test edge
    #[derive(Debug, Clone)]
    struct TestEdge {
        weight: f32,
    }

    impl ActivationEdge for TestEdge {
        fn weight(&self) -> f32 {
            self.weight
        }
    }

    fn create_test_graph() -> (DiGraph<TestNode, TestEdge>, HashMap<String, NodeIndex>) {
        let mut graph = DiGraph::new();
        let mut index = HashMap::new();

        // Create a simple graph: A -> B -> C -> D
        //                        A -> E
        let a = graph.add_node(TestNode {
            id: "A".to_string(),
        });
        let b = graph.add_node(TestNode {
            id: "B".to_string(),
        });
        let c = graph.add_node(TestNode {
            id: "C".to_string(),
        });
        let d = graph.add_node(TestNode {
            id: "D".to_string(),
        });
        let e = graph.add_node(TestNode {
            id: "E".to_string(),
        });

        graph.add_edge(a, b, TestEdge { weight: 1.0 });
        graph.add_edge(b, c, TestEdge { weight: 1.0 });
        graph.add_edge(c, d, TestEdge { weight: 1.0 });
        graph.add_edge(a, e, TestEdge { weight: 0.5 });

        index.insert("A".to_string(), a);
        index.insert("B".to_string(), b);
        index.insert("C".to_string(), c);
        index.insert("D".to_string(), d);
        index.insert("E".to_string(), e);

        (graph, index)
    }

    #[test]
    fn test_seed_activation() {
        let (graph, index) = create_test_graph();
        let config = SpreadingConfig::default();

        let results = spread_activation(&graph, &index, &[("A".to_string(), 1.0)], &config);

        // Seed should be in results with activation 1.0
        let seed = results.iter().find(|r| r.memory_id == "A").unwrap();
        assert!((seed.activation - 1.0).abs() < 0.01);
        assert_eq!(seed.depth, 0);
    }

    #[test]
    fn test_activation_decay() {
        let (graph, index) = create_test_graph();
        let config = SpreadingConfig {
            decay_factor: 0.5,
            firing_threshold: 0.01,
            max_depth: 4,
            fan_out_penalty: 0.0,
        };

        let results = spread_activation(&graph, &index, &[("A".to_string(), 1.0)], &config);

        // B should have activation ~0.5 (decay factor)
        let b = results.iter().find(|r| r.memory_id == "B");
        assert!(b.is_some());
        let b = b.unwrap();
        assert!(
            (b.activation - 0.5).abs() < 0.1,
            "B activation: {}",
            b.activation
        );
    }

    #[test]
    fn test_max_depth_limit() {
        let (graph, index) = create_test_graph();
        let config = SpreadingConfig {
            decay_factor: 0.9,
            firing_threshold: 0.01,
            max_depth: 2, // Only reach B, C, E from A
            fan_out_penalty: 0.0,
        };

        let results = spread_activation(&graph, &index, &[("A".to_string(), 1.0)], &config);

        // D should not be reached (depth 3)
        let d = results.iter().find(|r| r.memory_id == "D");
        assert!(d.is_none(), "D should not be reached at max_depth=2");
    }

    #[test]
    fn test_threshold_cutoff() {
        let (graph, index) = create_test_graph();
        let config = SpreadingConfig {
            decay_factor: 0.3,
            firing_threshold: 0.1, // High threshold
            max_depth: 10,
            fan_out_penalty: 0.0,
        };

        let results = spread_activation(&graph, &index, &[("A".to_string(), 1.0)], &config);

        // With decay=0.3, only A and B should exceed 0.1 threshold
        // A=1.0, B=0.3, C=0.09 (below)
        assert!(
            results.len() <= 3,
            "Should have limited results: {:?}",
            results
        );
    }

    #[test]
    fn test_edge_weight_modulation() {
        let (graph, index) = create_test_graph();
        let config = SpreadingConfig {
            decay_factor: 1.0, // No decay
            firing_threshold: 0.01,
            max_depth: 2,
            fan_out_penalty: 0.0,
        };

        let results = spread_activation(&graph, &index, &[("A".to_string(), 1.0)], &config);

        // E has edge weight 0.5, so activation should be ~0.5
        let e = results.iter().find(|r| r.memory_id == "E");
        assert!(e.is_some());
        let e = e.unwrap();
        assert!(e.activation < 0.6, "E activation: {}", e.activation);
    }

    #[test]
    fn test_spread_activation_by_id() {
        let (graph, index) = create_test_graph();
        let config = SpreadingConfig::default();

        let results = spread_activation_by_id(
            &graph,
            &index,
            &[("A".to_string(), 1.0)],
            &config,
            |node| node.id.clone(),
            |edge| edge.weight,
        );

        // Seed should be in results with activation 1.0
        let seed = results.iter().find(|r| r.memory_id == "A").unwrap();
        assert!((seed.activation - 1.0).abs() < 0.01);
        assert_eq!(seed.depth, 0);
    }
}
