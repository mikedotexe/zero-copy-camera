#[cfg(target_os = "macos")]
mod mac_app;

#[cfg(target_os = "macos")]
mod starlit_blossom_713;

#[cfg(target_os = "macos")]
fn main() {
    mac_app::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!(
        "⚠️  This zero-copy Metal demo currently runs only on macOS with a shared camera feed."
    );
}
