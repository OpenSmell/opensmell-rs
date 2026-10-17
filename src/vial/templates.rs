//! Embedded starters for `noses create`.
//!
//! Templates are compiled into the binary so `create` works fully offline — the
//! Tidyverse rule ("a package should scaffold in <60s") applies even with no
//! network. Every starter is a *valid* Vial: `noses lint` passes on it out of
//! the box, and TODO markers show where to go next.

use std::path::{Path, PathBuf};

/// Kebab-case a human name into a directory / id segment.
pub fn kebab(name: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

struct File {
    path: &'static str,
    body: &'static str,
}

const RECIPE_FILES: &[File] = &[
    File { path: "vial.toml", body: include_str!("../../vial/templates/recipe/vial.toml") },
    File { path: "recipe.json", body: include_str!("../../vial/templates/recipe/recipe.json") },
    File { path: "README.md", body: include_str!("../../vial/templates/recipe/README.md") },
    File { path: "fixtures/sample.csv", body: include_str!("../../vial/templates/recipe/fixtures/sample.csv") },
    File { path: "fixtures/expected.json", body: include_str!("../../vial/templates/recipe/fixtures/expected.json") },
];

const WASM_FILES: &[File] = &[
    File { path: "vial.toml", body: include_str!("../../vial/templates/wasm/vial.toml") },
    File { path: "Cargo.toml", body: include_str!("../../vial/templates/wasm/Cargo.toml") },
    File { path: "src/lib.rs", body: include_str!("../../vial/templates/wasm/src/lib.rs") },
    File { path: "wit/vial.wit", body: include_str!("../../vial/templates/wasm/wit/vial.wit") },
    File { path: "README.md", body: include_str!("../../vial/templates/wasm/README.md") },
];

/// Scaffold a new Vial. Returns the created directory. Refuses to clobber.
pub fn create(name: &str, template: &str, dest_root: &Path) -> Result<PathBuf, String> {
    if name.trim().is_empty() {
        return Err("a name is required: noses create <name>".into());
    }
    let slug = kebab(name);
    if slug.is_empty() {
        return Err("the name must contain at least one letter or digit".into());
    }

    let files = match template {
        "recipe" => RECIPE_FILES,
        "wasm" => WASM_FILES,
        other => return Err(format!("unknown template `{other}` (expected `recipe` or `wasm`)")),
    };

    let dir = dest_root.join(&slug);
    if dir.exists() {
        return Err(format!("{} already exists — refusing to overwrite", dir.display()));
    }

    let id = format!("org.example.{slug}");
    for f in files {
        let target = dir.join(f.path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let body = f
            .body
            .replace("__NAME__", name)
            .replace("__ID__", &id)
            .replace("__SLUG__", &slug);
        std::fs::write(&target, body).map_err(|e| format!("cannot write {}: {e}", target.display()))?;
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kebab_cases() {
        assert_eq!(kebab("Fermentation Monitor"), "fermentation-monitor");
        assert_eq!(kebab("  Beer!  Vial "), "beer-vial");
    }

    #[test]
    fn create_recipe_scaffolds_valid_vial() {
        let root = std::env::temp_dir().join("noses_create_test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let dir = create("My Vial", "recipe", &root).unwrap();
        assert!(dir.join("vial.toml").exists());
        assert!(dir.join("recipe.json").exists());
        let m = crate::vial::manifest::VialManifest::load(&dir.join("vial.toml")).unwrap();
        assert!(m.validate().iter().all(|d| d.level != crate::vial::manifest::Level::Error));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn example_vials_are_valid_and_fixtures_match() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vial/examples");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("examples/ missing").flatten() {
            let vd = entry.path();
            if !vd.is_dir() {
                continue;
            }
            let m = crate::vial::manifest::VialManifest::load(&vd.join("vial.toml")).unwrap();
            assert!(
                m.validate().iter().all(|d| d.level != crate::vial::manifest::Level::Error),
                "example {} has validation errors",
                vd.display()
            );
            let recipe: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(vd.join("recipe.json")).unwrap()).unwrap();
            let table = crate::vial::recipe::read_csv(&vd.join("fixtures/sample.csv")).unwrap();
            let out = crate::vial::recipe::evaluate(&recipe, &table).unwrap();
            let expected: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(vd.join("fixtures/expected.json")).unwrap()).unwrap();
            let problems = crate::vial::recipe::compare(&out, &expected, 1e-6);
            assert!(problems.is_empty(), "example {}: {problems:?}", vd.display());
            checked += 1;
        }
        assert!(checked >= 3, "expected at least 3 example Vials, found {checked}");
    }
}
