//! `noses` — author, validate, test and publish Osmograph Vials.
//!
//! One format: `vial.json` (`osmograph.domain-pack/1`) plus optional sandboxed
//! HTML fragments beside it. The loop is `new` → `lint` → `test` → `publish`.
//!
//! The desktop host is the authority: it re-runs the same gate when a manifest
//! is installed from a URL or a file, and the sandbox is the execution boundary.
//! This tool gives the author the answer before pushing; it never replaces the
//! host gate and it never signs or uploads anything.

use std::path::PathBuf;
use std::process::ExitCode;

use opensmell::vial::{scaffold, Vial};

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
        "help" | "-h" | "--help" => { print_help(); Ok(ExitCode::SUCCESS) }
        "version" | "-V" | "--version" => { println!("noses {VERSION}"); Ok(ExitCode::SUCCESS) }
        "new" | "create" => cmd_new(&args[1..]),
        "lint" | "check" | "validate" => cmd_lint(&args[1..]),
        "test" => cmd_test(&args[1..]),
        "publish" => cmd_publish(&args[1..]),
        other => Err(format!("unknown command `{other}` (try `noses help`)")),
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let inline = format!("--{name}=");
    if let Some(hit) = args.iter().find(|a| a.starts_with(&inline)) {
        return Some(hit[inline.len()..].to_string());
    }
    let i = args.iter().position(|a| a == &format!("--{name}"))?;
    args.get(i + 1).cloned()
}

fn positional(args: &[String]) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with('-') {
            if !a.contains('=') { i += 1; }
        } else {
            return Some(a.clone());
        }
        i += 1;
    }
    None
}

fn resolve(arg: Option<String>) -> PathBuf {
    let p = arg.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    if p.is_dir() { p.join("vial.json") } else { p }
}

fn cmd_new(args: &[String]) -> Result<ExitCode, String> {
    let name = positional(args).ok_or("usage: noses new <Name> [--publisher <handle>] [--domain <domain>]")?;
    let publisher = flag(args, "publisher").unwrap_or_else(|| "yourname".to_string());
    let domain = flag(args, "domain").unwrap_or_else(|| "generic".to_string());
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let dir = scaffold::create(&name, &publisher, &domain, &cwd)?;
    println!("Created {}", dir.display());
    println!();
    println!("Two gaps are deliberate — `noses lint` refuses until you fill them:");
    println!("  1. citations is empty: add the real references behind your protocol.");
    println!("  2. the text carries REPLACE ME markers: write what this Vial is and is not.");
    println!();
    println!("Next:");
    println!("  cd {} && noses lint", dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
    Ok(ExitCode::SUCCESS)
}

fn print_validation(vial: &Vial) -> bool {
    let v = vial.validate();
    for check in &v.checks {
        let mark = if check.passed { "✓" } else if check.critical { "✗" } else { "!" };
        println!("{mark} {:<22} {}", check.label, check.detail);
    }
    println!();
    if v.passed() {
        println!("✓ {} is valid ({} warning(s))", vial.id(), v.warnings.len());
        true
    } else {
        println!("✗ {} failed the gate ({} error(s), {} warning(s))", vial.id(), v.errors.len(), v.warnings.len());
        false
    }
}

fn cmd_lint(args: &[String]) -> Result<ExitCode, String> {
    let path = resolve(positional(args));
    let vial = Vial::load(&path)?;
    println!("Checking {}\n", vial.path.display());
    let ok = print_validation(&vial);
    Ok(if ok { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

fn cmd_test(args: &[String]) -> Result<ExitCode, String> {
    let path = resolve(positional(args));
    let vial = Vial::load(&path)?;
    println!("Testing {}\n", vial.path.display());
    let ok = print_validation(&vial);

    println!();
    println!("Programmable surface (static, no hardware)");
    let mut specs: Vec<(String, String)> = Vec::new();
    for key in ["ui", "homeUi"] {
        if let Some(entry) = vial.json.get(key).and_then(|s| s.get("entry")).and_then(|e| e.as_str()) {
            specs.push((key.to_string(), entry.to_string()));
        }
    }
    if specs.is_empty() {
        println!("  · no sandboxed UI declared; this Vial is manifest-only");
    }
    for (key, entry) in specs {
        let asset = vial.dir().join(&entry);
        let body = std::fs::read_to_string(&asset).unwrap_or_default();
        let has = |s: &str| body.contains(s);
        println!("  {key} → {entry}");
        println!("    onData           {}", yes_no(has("onData")));
        println!("    setTransforms    {}", yes_no(has("setTransforms")));
        println!("    setTrajectory    {}", yes_no(has("setTrajectory")));
        println!("    setDetection     {}", yes_no(has("setDetection")));
    }
    println!();
    println!("Static checks only. The live proof is the app: install this Vial, arm the");
    println!("watch on a connected rig, and confirm the host reports the transforms and");
    println!("the trajectory it actually applied.");
    Ok(if ok { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

fn yes_no(b: bool) -> &'static str {
    if b { "declared" } else { "—" }
}

fn cmd_publish(args: &[String]) -> Result<ExitCode, String> {
    let path = resolve(positional(args));
    let vial = Vial::load(&path)?;
    println!("Publishing {}\n", vial.path.display());
    let ok = print_validation(&vial);
    if !ok {
        return Err("fix the errors above before publishing".into());
    }
    let repo = flag(args, "repo").ok_or(
        "publish needs --repo https://github.com/<owner>/<repo> so the registry entry has real URLs")?;
    let branch = flag(args, "branch").unwrap_or_else(|| "main".to_string());
    let evidence = flag(args, "evidence");

    let publisher = vial.id().split('.').next().unwrap_or("");
    if ["yourname", "you", "author", "example"].contains(&publisher) {
        return Err(format!(
            "`{}` is a placeholder publisher; set a real id in vial.json before publishing",
            publisher
        ));
    }
    if vial.has_ui() {
        println!();
        println!("This Vial ships a sandboxed UI. It must be installed from its manifest URL:");
        println!("push the files first, then confirm the raw URL below serves vial.json AND the");
        println!("HTML fragments beside it.");
    }

    let entry = vial.registry_entry(&repo, &branch, evidence.as_deref())?;
    println!();
    println!("sha256(vial.json) = {}", vial.sha256_hex());
    println!();
    println!("Registry entry (add to public/vials/registry.json in OpenSmell/OpenSmell-web):");
    println!("{}", serde_json::to_string_pretty(&entry).map_err(|e| e.to_string())?);
    println!();
    println!("Steps:");
    println!("  1. push this directory to {repo} (the manifest must be at the repo root);");
    println!("  2. verify the raw manifest URL returns exactly these bytes:");
    println!("     {}", entry["manifestUrl"].as_str().unwrap_or(""));
    println!("  3. open a PR that adds the entry above. A listing is discovery, not a review:");
    println!("     every install re-runs this gate and every UI runs in the same sandbox.");
    Ok(ExitCode::SUCCESS)
}

fn print_help() {
    println!(
        "noses {VERSION} — author, validate, test and publish Osmograph Vials\n\
\n\
USAGE:\n\
  noses new <Name> [--publisher <handle>] [--domain <domain>]  scaffold a real Vial\n\
  noses lint [path]                                            the policy gate, locally\n\
  noses test [path]                                            lint + what the UI declares\n\
  noses publish [path] --repo <github-url> [--branch main]     hash + registry entry\n\
  noses help | version\n\
\n\
A Vial is one `vial.json` (osmograph.domain-pack/1) plus optional sandboxed\n\
HTML fragments beside it. The desktop host re-runs the same gate at install and\n\
is the authority. Nothing here signs or uploads anything.\n\
\n\
Docs + LLM index: https://opensmell.org/llms.txt"
    );
}
