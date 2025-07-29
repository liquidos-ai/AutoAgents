use std::{env, fs, path::Path};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = Path::new(&manifest_dir).join("wasm");

    // Ensure the wasm output directory exists
    fs::create_dir_all(&out_dir).expect("Failed to create wasm output directory");

    let wasm_name = "wasm_tool.wasm";

    let target_wasm_path = Path::new(&manifest_dir)
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join(&wasm_name);

    if target_wasm_path.exists() {
        let out_wasm_path = out_dir.join(&wasm_name);

        fs::copy(&target_wasm_path, &out_wasm_path)
            .expect("Failed to copy wasm file to wasm/ directory");

        println!("cargo:warning=Copied {} to wasm/", wasm_name);
    } else {
        println!("cargo:warning=WASM file not found. Build it manually:");
    }

    // Re-run build.rs if these files change
    println!("cargo:rerun-if-changed=src/lib.rs");
}
