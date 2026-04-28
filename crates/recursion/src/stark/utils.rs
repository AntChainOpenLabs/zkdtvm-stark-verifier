pub fn dt_dev_mode() -> bool {
    let value = std::env::var("DT_DEV").unwrap_or_else(|_| "false".to_string());
    let enabled = value == "1" || value.to_lowercase() == "true";
    if enabled {
        tracing::warn!("DT_DEV environment variable is enabled. do not enable this in production");
    }
    enabled
}
