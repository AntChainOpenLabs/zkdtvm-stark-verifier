use std::{
    io::Write,
    path::{Path, PathBuf},
};

use dt_primitives::SCField;
use dt_stark::sumcheck::trace::CompressedMatrix;

/// The ledger directory, when instrumentation is enabled.
pub fn ledger_dir() -> Option<PathBuf> {
    std::env::var("DT_RECURSION_LEDGER_DIR").ok().map(PathBuf::from)
}

/// Appends one JSON value as a line to `<dir>/<file>` (best-effort: instrumentation
/// must never fail the prove).
pub fn append(dir: &Path, file: &str, value: &serde_json::Value) {
    let path = dir.join(file);
    let result = std::fs::create_dir_all(dir).and_then(|()| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut handle| writeln!(handle, "{value}"))
    });
    if let Err(err) = result {
        tracing::warn!("stage ledger append to {} failed: {err}", path.display());
    }
}

/// Per-chip shape rows from generated traces: (chip, stored rows, padded rows, width).
pub fn chip_rows(traces: &[(String, CompressedMatrix<SCField>)]) -> serde_json::Value {
    serde_json::Value::Array(
        traces
            .iter()
            .map(|(name, trace)| {
                serde_json::json!({
                    "chip": name,
                    "stored_height": p3_matrix::Matrix::height(&trace.main),
                    "padded_height": trace.total_height,
                    "width": p3_matrix::Matrix::width(&trace.main),
                })
            })
            .collect(),
    )
}

/// Linux VmHWM (peak resident set) in kB; `None` where /proc is absent.
pub fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1).and_then(|value| value.parse().ok()))
}
