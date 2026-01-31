//! Strength signal endpoints for memory feedback.

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use rook_core::{Grade, StrengthSignal};

/// Request body for processing strength signals.
#[derive(Debug, Deserialize)]
pub struct ProcessSignalsRequest {
    /// The signals to process.
    pub signals: Vec<SignalInput>,
}

/// A strength signal input from the API.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SignalInput {
    /// Memory was used in generating a response.
    UsedInResponse {
        memory_id: String,
        #[serde(default)]
        context: Option<String>,
    },
    /// User explicitly corrected information.
    UserCorrection {
        old_memory_id: String,
        new_content: String,
    },
    /// User confirmed information is correct.
    UserConfirmation { memory_id: String },
    /// Contradiction resolved between two memories.
    Contradiction { winner_id: String, loser_id: String },
    /// Memory was retrieved but not used.
    RetrievedNotUsed { memory_id: String },
    /// User explicitly marked memory as incorrect.
    MarkedIncorrect {
        memory_id: String,
        #[serde(default)]
        reason: Option<String>,
    },
    /// User explicitly marked memory as important.
    MarkedImportant { memory_id: String },
}

impl From<SignalInput> for StrengthSignal {
    fn from(input: SignalInput) -> Self {
        match input {
            SignalInput::UsedInResponse { memory_id, context } => {
                StrengthSignal::UsedInResponse { memory_id, context }
            }
            SignalInput::UserCorrection {
                old_memory_id,
                new_content,
            } => StrengthSignal::UserCorrection {
                old_memory_id,
                new_content,
            },
            SignalInput::UserConfirmation { memory_id } => {
                StrengthSignal::UserConfirmation { memory_id }
            }
            SignalInput::Contradiction {
                winner_id,
                loser_id,
            } => StrengthSignal::Contradiction {
                winner_id,
                loser_id,
            },
            SignalInput::RetrievedNotUsed { memory_id } => {
                StrengthSignal::RetrievedNotUsed { memory_id }
            }
            SignalInput::MarkedIncorrect { memory_id, reason } => {
                StrengthSignal::MarkedIncorrect { memory_id, reason }
            }
            SignalInput::MarkedImportant { memory_id } => {
                StrengthSignal::MarkedImportant { memory_id }
            }
        }
    }
}

/// Response for processing signals.
#[derive(Debug, Serialize)]
pub struct ProcessSignalsResponse {
    /// Number of signals processed.
    pub processed: usize,
    /// Grade updates pending application.
    pub pending_updates: Vec<PendingUpdate>,
    /// Message.
    pub message: String,
}

/// A pending grade update.
#[derive(Debug, Serialize)]
pub struct PendingUpdate {
    pub memory_id: String,
    pub grade: String,
}

/// Process strength signals.
/// POST /signals
pub async fn process_signals(
    State(state): State<AppState>,
    Json(request): Json<ProcessSignalsRequest>,
) -> ApiResult<Json<ProcessSignalsResponse>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    let signal_count = request.signals.len();

    // Process each signal
    {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        for signal_input in request.signals {
            let signal: StrengthSignal = signal_input.into();
            memory.process_strength_signal(signal);
        }
    }

    // Get pending updates
    let pending_updates = {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory
            .get_pending_strength_updates()
            .into_iter()
            .map(|(memory_id, grade)| PendingUpdate {
                memory_id,
                grade: grade_to_string(grade),
            })
            .collect()
    };

    Ok(Json(ProcessSignalsResponse {
        processed: signal_count,
        pending_updates,
        message: format!("Processed {} signal(s)", signal_count),
    }))
}

fn grade_to_string(grade: Grade) -> String {
    match grade {
        Grade::Again => "again".to_string(),
        Grade::Hard => "hard".to_string(),
        Grade::Good => "good".to_string(),
        Grade::Easy => "easy".to_string(),
    }
}

/// Request to apply pending updates (clear the processor).
#[derive(Debug, Deserialize)]
pub struct ApplyUpdatesRequest {
    /// Whether to clear pending updates after returning them.
    #[serde(default = "default_clear")]
    pub clear: bool,
}

fn default_clear() -> bool {
    true
}

/// Response with applied updates.
#[derive(Debug, Serialize)]
pub struct ApplyUpdatesResponse {
    /// Updates that were applied.
    pub updates: Vec<PendingUpdate>,
    /// Whether updates were cleared.
    pub cleared: bool,
}

/// Get and optionally clear pending strength updates.
/// POST /signals/apply
pub async fn apply_updates(
    State(state): State<AppState>,
    Json(request): Json<ApplyUpdatesRequest>,
) -> ApiResult<Json<ApplyUpdatesResponse>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    let guard = state.inner.read().await;
    let memory = guard
        .memory
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

    let updates: Vec<PendingUpdate> = memory
        .get_pending_strength_updates()
        .into_iter()
        .map(|(memory_id, grade)| PendingUpdate {
            memory_id,
            grade: grade_to_string(grade),
        })
        .collect();

    if request.clear {
        memory.clear_strength_updates();
    }

    Ok(Json(ApplyUpdatesResponse {
        updates,
        cleared: request.clear,
    }))
}
