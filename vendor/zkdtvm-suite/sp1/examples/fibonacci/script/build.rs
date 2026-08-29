fn main() {
    let mut args = dt_build::guest_program_build_args();
    args.ignore_rust_version = true;
    dt_build::build_program_with_args("../program", args);
}
