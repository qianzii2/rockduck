fn main() {
    // Fix Windows linking for DuckDB bundled feature
    // DuckDB uses Windows Restart Manager API
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=dylib=rstrtmgr");
        println!("cargo:rustc-link-lib=dylib=ole32");
    }
}
