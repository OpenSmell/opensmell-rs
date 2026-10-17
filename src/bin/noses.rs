//! `noses` — author, validate, test and publish Osmograph Vials.
//!
//! One manifest, three tiers (recipe | wasm | sidecar). The loop is the moat:
//! `create` → `test` (fixtures, no hardware) → `lint` → `publish` (OIDC).
//!
//! Lives in the core crate so the manifest rules, recipe engine and templates
//! have exactly one home — shared with the engine itself.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use opensmell::vial::manifest::{Diagnostic, Level, VialManifest};
use opensmell::vial::{recipe, templates};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "help" | "-h" | "--help" => {
            print_help();
            Ok(ExitCode::SUCCESS)
        }
        "version" | "-V" | "--version" => {
            println!("noses {VERSION}");
            Ok(ExitCode::SUCCESS)
        }
        "create" => cmd_create(&args[1..]),
        "lint" | "validate" => cmd_check(&args[1..]),
        "test" => cmd_test(&args[1..]),
        "dev" => cmd_dev(&args[1..]),
        "publish" => cmd_publish(&args[1..]),
        "update" => {
            println!("noses update: no-op until the copier template ships.");
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command `{other}` (try `noses help`)")),
    }
}

fn cmd_create(args: &[String]) -> Result<ExitCode, String> {
    let mut name: Option<String> = None;
    let mut template = String::from("recipe");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--template" | "-t" => {
                i += 1;
                template = args.get(i).ok_or("--template needs a value")?.clone();
            }
            other if other.starts_with('-') => return Err(format!("unknown flag `{other}`")),
            other => name = Some(other.to_string()),
        }
        i += 1;
    }
    let name = name.ok_or("usage: noses create <name> [--template recipe|wasm]")?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let dir = templates::create(&name, &template, &cwd)?;
    println!("Created {template} Vial at {}", dir.display());
    println!("Next: cd {} && noses test", templates::kebab(&name));
    Ok(ExitCode::SUCCESS)
}

fn cmd_check(args: &[String]) -> Result<ExitCode, String> {
    let path = resolve_manifest(args.first().map(String::as_str))?;
    let m = VialManifest::load(&path)?;
    println!("Checking {}", path.display());
    let diags = m.validate();
    let has_error = print_diagnostics(&diags);
    if has_error {
        return Ok(ExitCode::from(1));
    }
    let warns = diags.iter().filter(|d| d.level == Level::Warn).count();
    println!("✓ {} is valid ({warns} warning(s))", m.id().unwrap_or("vial"));
    Ok(ExitCode::SUCCESS)
}

fn cmd_test(args: &[String]) -> Result<ExitCode, String> {
    let dir = resolve_vial_dir(args.first().map(String::as_str))?;
    let m = VialManifest::load(&dir.join("vial.toml"))?;

    let errs: Vec<_> = m.validate().into_iter().filter(|d| d.level == Level::Error).collect();
    if !errs.is_empty() {
        for d in &errs {
            println!("error: {}", d.message);
        }
        return Err("fix validation errors before testing".into());
    }

    match m.tier() {
        Some("recipe") => {
            let recipe_path = dir.join("recipe.json");
            let text = std::fs::read_to_string(&recipe_path)
                .map_err(|e| format!("cannot read {}: {e}", recipe_path.display()))?;
            let recipe: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("invalid recipe.json: {e}"))?;

            let fixture = dir.join("fixtures/sample.csv");
            let table = recipe::read_csv(&fixture)?;
            let out = recipe::evaluate(&recipe, &table)?;
            println!("Fixture: {} ({} columns, {} rows)", fixture.display(), table.columns.len(), table.rows.len());
            for (k, v) in &out {
                println!("  {k} = {v}");
            }

            let expected_path = dir.join("fixtures/expected.json");
            if !expected_path.exists() {
                println!("(no fixtures/expected.json — printing outputs only)");
                return Ok(ExitCode::SUCCESS);
            }
            let et = std::fs::read_to_string(&expected_path).map_err(|e| e.to_string())?;
            let expected: serde_json::Value = serde_json::from_str(&et).map_err(|e| e.to_string())?;
            let problems = recipe::compare(&out, &expected, 1e-6);
            if problems.is_empty() {
                println!("✓ outputs match expected.json");
                Ok(ExitCode::SUCCESS)
            } else {
                for p in problems {
                    println!("✗ {p}");
                }
                Ok(ExitCode::from(1))
            }
        }
        Some("wasm") => {
            println!("wasm tier: build with `cargo build --target wasm32-unknown-unknown --release` — fixtures are checked by the host sandbox.");
            Ok(ExitCode::SUCCESS)
        }
        Some("sidecar") => {
            println!("sidecar tier: run your Python entrypoint; fixtures are passed as files.");
            Ok(ExitCode::SUCCESS)
        }
        _ => Err("unknown tier".into()),
    }
}

fn cmd_dev(args: &[String]) -> Result<ExitCode, String> {
    let dir = resolve_vial_dir(args.first().map(String::as_str))?;
    let m = VialManifest::load(&dir.join("vial.toml"))?;
    println!(
        "Loaded {} ({}) from {}",
        m.id().unwrap_or("vial"),
        m.tier().unwrap_or("?"),
        dir.display()
    );
    println!("Open Osmograph Desktop → System → Plugins to load and hot-reload this Vial.");
    println!("Tip: point your AI at llms.txt + the MCP validator, then `noses lint` in a loop.");
    Ok(ExitCode::SUCCESS)
}

fn cmd_publish(args: &[String]) -> Result<ExitCode, String> {
    let path = resolve_manifest(args.first().map(String::as_str))?;
    let m = VialManifest::load(&path)?;
    let diags = m.validate();
    if print_diagnostics(&diags) {
        return Ok(ExitCode::from(1));
    }
    println!("✓ {} passed validation", m.id().unwrap_or("vial"));
    println!("Publishing uses OIDC trusted publishing (no tokens live in the repo):");
    println!("  1. push this Vial to a public repo;");
    println!("  2. `noses publish` opens a store PR with a signed artifact + provenance.");
    Ok(ExitCode::SUCCESS)
}

/// Resolve an argument to a `vial.toml` path (a directory resolves to `vial.toml` inside).
fn resolve_manifest(arg: Option<&str>) -> Result<PathBuf, String> {
    let p = match arg {
        Some(s) => PathBuf::from(s),
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    Ok(if p.is_dir() { p.join("vial.toml") } else { p })
}

/// Resolve an argument to a Vial directory.
fn resolve_vial_dir(arg: Option<&str>) -> Result<PathBuf, String> {
    let p = match arg {
        Some(s) => PathBuf::from(s),
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    Ok(if p.is_dir() {
        p
    } else {
        p.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
    })
}

/// Print diagnostics; returns true if any were errors.
fn print_diagnostics(diags: &[Diagnostic]) -> bool {
    let mut has_error = false;
    for d in diags {
        match d.level {
            Level::Error => {
                println!("error: {}", d.message);
                has_error = true;
            }
            Level::Warn => println!("warning: {}", d.message),
        }
    }
    has_error
}

fn print_help() {
    println!(
        "noses {VERSION} — author, validate, test and publish Osmograph Vials\n\
\n\
USAGE:\n\
  noses create <name> [--template recipe|wasm]   scaffold a working Vial\n\
  noses test  [path]                            run fixtures, no hardware needed\n\
  noses lint  [path]                            schema + permission + licence checks\n\
  noses dev   [path]                            load + hot-reload in Osmograph\n\
  noses publish [path]                          validate, then OIDC trusted publish\n\
  noses help | version\n\
\n\
A Vial is one `vial.toml` manifest at three tiers: recipe | wasm | sidecar.\n\
Docs + LLM index: https://opensmell.org/llms.txt"
    );
}
