use std::{env, fs, path::Path};

const FILENAME: &str = "vk_map.bin";

fn main() {
    println!("cargo:rerun-if-env-changed=VK_MAP_SRC_PATH");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    let out_path = out_dir.join(FILENAME);

    if let Ok(src_path) = env::var("VK_MAP_SRC_PATH") {
        let src = Path::new(&src_path);
        if src.exists() {
            fs::copy(src, &out_path).expect("failed to copy VK map");
            return;
        }
    }

    let src_path = Path::new("src").join(FILENAME);
    if src_path.exists() {
        fs::copy(&src_path, &out_path).expect("failed to copy VK map from src");
        return;
    }

    if !out_path.exists() {
        use std::collections::BTreeMap;
        let empty_map: BTreeMap<[u32; 8], usize> = BTreeMap::new();
        let data = bincode::serialize(&empty_map).unwrap();
        fs::write(&out_path, data).unwrap();
    }
}
