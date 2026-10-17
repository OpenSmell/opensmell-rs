//! A tiny, dependency-light recipe engine used by `noses test`.
//!
//! A tier-1 Vial describes its work declaratively in `recipe.json`: a list of
//! steps over CSV columns. This engine evaluates those steps against a fixture
//! recording so an author can iterate **without hardware** — the whole point of
//! the dev loop.
//!
//! Supported ops (kept deliberately small; real transforms live in the Vial
//! runtime, this is the authoring/reference implementation):
//!   mean, min, max, range, first, last, std (population), delta(a - b)

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Parsed CSV: a header row and numeric rows.
pub struct Table {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

impl Table {
    pub fn column(&self, name: &str) -> Result<Vec<f64>, String> {
        let idx = self
            .columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| format!("column `{name}` not found (have: {})", self.columns.join(", ")))?;
        Ok(self.rows.iter().map(|r| r[idx]).collect())
    }
}

/// Read a numeric CSV. The first non-empty line is the header.
pub fn read_csv(path: &Path) -> Result<Table, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| format!("{} is empty", path.display()))?;
    let columns: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let cells: Vec<f64> = line
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|_| format!("{}: row {} has a non-numeric cell `{}`", path.display(), i + 2, s.trim()))
            })
            .collect::<Result<_, _>>()?;
        if cells.len() != columns.len() {
            return Err(format!(
                "{}: row {} has {} cells but the header has {}",
                path.display(),
                i + 2,
                cells.len(),
                columns.len()
            ));
        }
        rows.push(cells);
    }
    if rows.is_empty() {
        return Err(format!("{} has a header but no data rows", path.display()));
    }
    Ok(Table { columns, rows })
}

/// Evaluate a recipe (`{ "steps": [ { "op": ..., "as": ... } ] }`) over a table.
pub fn evaluate(recipe: &Value, table: &Table) -> Result<BTreeMap<String, f64>, String> {
    let steps = recipe
        .get("steps")
        .and_then(Value::as_array)
        .ok_or_else(|| "recipe.json must contain a `steps` array".to_string())?;

    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    for (i, step) in steps.iter().enumerate() {
        let op = step.get("op").and_then(Value::as_str).ok_or_else(|| format!("step {i} is missing `op`"))?;
        let name = step
            .get("as")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("step {i} (`{op}`) is missing `as`"))?
            .to_string();

        let value = match op {
            "mean" | "min" | "max" | "range" | "first" | "last" | "std" => {
                let channel = step
                    .get("channel")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("step {i} (`{op}`) is missing `channel`"))?;
                let data = table.column(channel)?;
                match op {
                    "mean" => mean(&data),
                    "min" => data.iter().cloned().fold(f64::INFINITY, f64::min),
                    "max" => data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    "range" => {
                        let hi = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let lo = data.iter().cloned().fold(f64::INFINITY, f64::min);
                        hi - lo
                    }
                    "first" => *data.first().unwrap(),
                    "last" => *data.last().unwrap(),
                    "std" => std_dev(&data),
                    _ => unreachable!(),
                }
            }
            "delta" => {
                let a = resolve(step, "a", &out)?;
                let b = resolve(step, "b", &out)?;
                a - b
            }
            other => return Err(format!("step {i}: unknown op `{other}`")),
        };
        out.insert(name, value);
    }
    Ok(out)
}

fn resolve(step: &Value, key: &str, out: &BTreeMap<String, f64>) -> Result<f64, String> {
    let name = step
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("delta is missing `{key}`"))?;
    out.get(name).copied().ok_or_else(|| format!("delta references `{name}` before it is defined"))
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    let m = mean(data);
    let var = data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}

/// Compare computed outputs against an `expected.json` object. Returns a list of
/// human-readable mismatches (empty = match).
pub fn compare(actual: &BTreeMap<String, f64>, expected: &Value, tol: f64) -> Vec<String> {
    let mut problems = Vec::new();
    let obj = match expected.as_object() {
        Some(o) => o,
        None => return vec!["expected.json must be a JSON object of name → number".to_string()],
    };
    for (name, want) in obj {
        let want = match want.as_f64() {
            Some(v) => v,
            None => {
                problems.push(format!("expected.{name} is not a number"));
                continue;
            }
        };
        match actual.get(name) {
            Some(got) if (got - want).abs() <= tol => {}
            Some(got) => problems.push(format!("{name}: expected {want}, got {got}")),
            None => problems.push(format!("{name}: missing from recipe output")),
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Table {
        Table {
            columns: vec!["timestamp_ms".into(), "ch0".into()],
            rows: vec![vec![0.0, 100.0], vec![1000.0, 110.0], vec![2000.0, 120.0], vec![3000.0, 105.0], vec![4000.0, 130.0]],
        }
    }

    #[test]
    fn computes_stats_and_delta() {
        let recipe: Value = serde_json::json!({
            "steps": [
                { "op": "mean", "channel": "ch0", "as": "m" },
                { "op": "min", "channel": "ch0", "as": "lo" },
                { "op": "max", "channel": "ch0", "as": "hi" },
                { "op": "delta", "a": "hi", "b": "lo", "as": "span" }
            ]
        });
        let out = evaluate(&recipe, &table()).unwrap();
        assert_eq!(out["m"], 113.0);
        assert_eq!(out["span"], 30.0);
    }

    #[test]
    fn compare_reports_mismatch() {
        let mut actual = BTreeMap::new();
        actual.insert("x".to_string(), 2.0);
        let expected = serde_json::json!({ "x": 3.0 });
        assert_eq!(compare(&actual, &expected, 1e-9).len(), 1);
        let expected_ok = serde_json::json!({ "x": 2.0 });
        assert!(compare(&actual, &expected_ok, 1e-9).is_empty());
    }
}
