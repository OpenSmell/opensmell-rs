//! The Vial system: author, validate, test and ship Osmograph extensions.
//!
//! One manifest (`vial.toml`) at three tiers — recipe | wasm | sidecar — driven
//! by the `noses` toolchain (`src/bin/noses.rs`). Kept in the core crate so the
//! manifest rules, recipe engine and provenance model have exactly one home.

pub mod manifest;
pub mod recipe;
pub mod templates;
