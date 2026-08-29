fn main() {
    println!("cargo::rustc-check-cfg=cfg(dt_ci_in_progress)");
    if std::env::var("DT_CI_IN_PROGRESS").is_ok() {
        println!("cargo::rustc-cfg=dt_ci_in_progress");
    }
}
