//! # ternary-entropy-cli
//!
//! CLI tool that reads ternary sequences and computes entropy metrics.
//!
//! Reads a ternary sequence from stdin or a file, where symbols are
//! whitespace-separated integers from {-1, 0, +1}, and computes various
//! information-theoretic measures.

use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;

use clap::Parser;

use ternary_entropy::*;

/// CLI tool for computing entropy metrics on ternary sequences.
#[derive(Parser, Debug)]
#[command(name = "ternary-entropy-cli")]
#[command(about = "Compute entropy metrics for ternary sequences", long_about = None)]
struct Cli {
    /// Input file (default: stdin)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Window size for sliding-window entropy
    #[arg(short = 'w', long, default_value_t = 0)]
    window: usize,

    /// Entropy base: "nats" (default), "bits", "trits"
    #[arg(short = 'b', long, default_value = "nats")]
    base: String,

    /// Output raw sequence values (for sliding window only)
    #[arg(short = 'v', long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Read the sequence
    let sequence = if let Some(path) = &cli.input {
        read_sequence_from_file(path)?
    } else {
        read_sequence_from_stdin()?
    };

    if sequence.is_empty() {
        eprintln!("Error: empty sequence (no ternary symbols found)");
        std::process::exit(1);
    }

    // Validate all symbols are in {-1, 0, +1}
    for &s in &sequence {
        if !matches!(s, -1 | 0 | 1) {
            eprintln!("Error: invalid ternary symbol '{}'. Must be -1, 0, or +1.", s);
            std::process::exit(1);
        }
    }

    // Compute empirical distribution
    let total = sequence.len() as f64;
    let mut counts = [0usize; 3];
    for &s in &sequence {
        match s {
            -1 => counts[0] += 1,
            0 => counts[1] += 1,
            1 => counts[2] += 1,
            _ => unreachable!(),
        }
    }
    let probs = [
        counts[0] as f64 / total,
        counts[1] as f64 / total,
        counts[2] as f64 / total,
    ];

    let (entropy_fn, unit) = match cli.base.as_str() {
        "bits" => (ternary_shannon_entropy_bits as fn([f64; 3]) -> f64, "bits"),
        "trits" => (ternary_shannon_entropy_trits as fn([f64; 3]) -> f64, "trits"),
        _ => (ternary_shannon_entropy as fn([f64; 3]) -> f64, "nats"),
    };

    // Print results
    if cli.verbose {
        println!("=== Ternary Entropy Analysis ===\n");

        println!("Symbol counts:");

        if cli.verbose {
            println!("  -1: {} ({:.4}%)", counts[0], 100.0 * probs[0]);
            println!("   0: {} ({:.4}%)", counts[1], 100.0 * probs[1]);
            println!("  +1: {} ({:.4}%)", counts[2], 100.0 * probs[2]);
        } else {
            println!("  -1: {}", counts[0]);
            println!("   0: {}", counts[1]);
            println!("  +1: {}", counts[2]);
        }
        println!("  Total: {}", sequence.len());
        println!();
    }

    let h = entropy_fn(probs);
    println!("H(X) = {:.8} {}", h, unit);

    // Sliding window entropy
    if cli.window > 0 {
        let window_entropies = sliding_window_entropy(&sequence, cli.window);
        if window_entropies.is_empty() {
            eprintln!("Warning: window size {} is larger than sequence length {}", cli.window, sequence.len());
        } else {
            println!(
                "\nSliding window entropy (window={}, {}):",
                cli.window, unit
            );
            if cli.verbose {
                for (i, &h_win) in window_entropies.iter().enumerate() {
                    let window_vals: Vec<String> = sequence[i..i + cli.window]
                        .iter()
                        .map(|s| {
                            if *s == -1 {
                                "-".to_string()
                            } else {
                                s.to_string()
                            }
                        })
                        .collect();
                    println!(
                        "  [{}] {} -> {:.6} {}",
                        i,
                        window_vals.join(" "),
                        h_win,
                        unit
                    );
                }
            } else {
                for (i, &h_win) in window_entropies.iter().enumerate() {
                    println!("  [{}] {:.6}", i, h_win);
                }
            }
        }
    }

    Ok(())
}

/// Read a ternary sequence from a file, one symbol per line or whitespace-separated.
fn read_sequence_from_file(path: &PathBuf) -> Result<Vec<i8>, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let sequence = parse_ternary_string(&contents)?;
    Ok(sequence)
}

/// Read a ternary sequence from stdin.
fn read_sequence_from_stdin() -> Result<Vec<i8>, Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut sequence = Vec::new();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let more = parse_ternary_string(&line)?;
        sequence.extend(more);
    }
    Ok(sequence)
}

/// Parse whitespace-separated ternary symbols.
fn parse_ternary_string(s: &str) -> Result<Vec<i8>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    for token in s.split_whitespace() {
        let val: i8 = token.parse()?;
        if !matches!(val, -1 | 0 | 1) {
            return Err(format!("invalid ternary symbol: {}", val).into());
        }
        result.push(val);
    }
    Ok(result)
}
