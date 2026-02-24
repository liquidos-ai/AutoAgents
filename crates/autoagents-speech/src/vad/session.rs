use super::error::{VadError, VadResult};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;

pub fn create_session(path: &Path) -> VadResult<Session> {
    let session = Session::builder()
        .map_err(|err| VadError::ModelLoad(err.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| VadError::ModelLoad(err.to_string()))?
        .with_intra_threads(1)
        .map_err(|err| VadError::ModelLoad(err.to_string()))?
        .with_inter_threads(1)
        .map_err(|err| VadError::ModelLoad(err.to_string()))?
        .commit_from_file(path)
        .map_err(|err| VadError::ModelLoad(err.to_string()))?;
    Ok(session)
}
