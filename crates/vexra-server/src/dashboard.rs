//! Embedded web dashboard — served directly from the binary.
//!
//! A self-contained HTML/JS dashboard for managing EmbedDB databases.
//! Served at `/` and communicates with the REST API at `/api/*`.

/// The complete dashboard HTML page.
pub fn dashboard_html() -> String {
    include_str!("dashboard.html").to_string()
}
