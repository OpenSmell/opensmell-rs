//! The Vial system: author, validate, test and ship Osmograph extensions.
//!
//! One format — `vial.json` (`osmograph.domain-pack/1`) plus optional sandboxed
//! HTML fragments beside it. `noses` (`src/bin/noses.rs`) is the authoring tool;
//! the desktop host is the authority that re-runs the same gate at install time.

pub mod pack;
pub mod scaffold;

pub use pack::{Check, Validation, Vial, BRIDGE_METHODS, UI_MAX_BYTES};
