pub fn init_log() {
    // init pretty_env_logger
    pretty_env_logger::try_init().unwrap_or_default();
}
