use std::env;

use serde::{Deserialize, Serialize};
use sysinfo::System;
const MAX_SHARD_SIZE: usize = 1 << 26;
const RECURSION_MAX_SHARD_SIZE: usize = 1 << 22;
const MAX_SHARD_BATCH_SIZE: usize = 8;
const DEFAULT_TRACE_GEN_WORKERS: usize = 1;
const DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY: usize = 128;
const DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY: usize = 1;
const MAX_DEFERRED_SPLIT_THRESHOLD: usize = 1 << 15;
/// The maximum padded height allowed for any single chip within a shard.
pub const SHARD_HEIGHT_THRESHOLD: u64 = 1 << 22;

/// The maximum total cells (sum of `padded_height` * width across all chips) allowed per shard.
///
/// Basefold PCS has effective blowup factor 1 (vs FRI's 2), so we can use 1<<29 instead of
/// SP1's (1<<28)+(1<<27) while keeping the same memory footprint.
pub const SHARD_CELLS_THRESHOLD: u64 = (1u64 << 29) * 8 / 5;

/// A smaller total cells threshold for memory-constrained environments.
pub const SHARD_CELLS_THRESHOLD_SMALL: u64 = (1 << 28) + (1 << 27);

/// Independent cells budget for precompile shards. 1.5× the core threshold because
/// precompile shards have simpler structure (no CPU chip) and Basefold blowup=1 gives
/// us headroom. = (1<<29) + (1<<28) ≈ 805M cells.
pub const PRECOMPILE_SHARD_CELLS_THRESHOLD: u64 =
    SHARD_CELLS_THRESHOLD + (SHARD_CELLS_THRESHOLD >> 1);
/// The threshold that determines when to split the shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardingThreshold {
    /// The maximum number of elements in the trace.
    pub element_threshold: u64,
    /// The maximum number of rows for a single operation.
    pub height_threshold: u64,
}
/// The engine `DTProver::compress` dispatches to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionBackend {
    /// The native recursion ladder (koalabear + ext5 only; requires the
    /// `native-recursion` feature on the prover crate).
    Native,
    /// The DSL-compiled recursion pipeline.
    Dsl,
}

impl RecursionBackend {
    /// Resolve the effective backend: an explicit opts choice wins, else the
    /// `DT_RECURSION_BACKEND` env selector ("native" | "dsl"), else Native.
    /// An unparseable env value is an error, never a silent default.
    pub fn resolve(explicit: Option<Self>) -> Result<Self, String> {
        if let Some(backend) = explicit {
            return Ok(backend);
        }
        match env::var("DT_RECURSION_BACKEND") {
            Ok(value) => match value.to_ascii_lowercase().as_str() {
                "native" => Ok(Self::Native),
                "dsl" => Ok(Self::Dsl),
                other => {
                    Err(format!("DT_RECURSION_BACKEND must be 'native' or 'dsl', got '{other}'"))
                }
            },
            Err(_) => Ok(Self::Native),
        }
    }
}

/// Options to configure the DT prover for core and recursive proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DTProverOpts {
    /// Options for the core prover.
    pub core_opts: DTCoreOpts,
    /// Options for the recursion prover.
    pub recursion_opts: DTCoreOpts,
    /// Which recursion backend `compress` runs; `None` defers to the
    /// `DT_RECURSION_BACKEND` env selector (unset = native).
    #[serde(default)]
    pub recursion_backend: Option<RecursionBackend>,
}

impl DTProverOpts {
    /// Get the default prover options.
    #[must_use]
    pub fn auto() -> Self {
        let cpu_ram_gb = System::new_all().total_memory() / (1024 * 1024 * 1024);
        DTProverOpts::cpu(cpu_ram_gb as usize)
    }

    /// Get the memory options (shard size, shard batch size, and divisor) for a prover on CPU based
    /// on the amount of CPU memory.
    #[must_use]
    fn get_memory_opts(cpu_ram_gb: usize) -> (usize, usize, usize) {
        match cpu_ram_gb {
            0..33 => (22, 1, 3),
            33..49 => (23, 1, 2),
            49..65 => (24, 1, 3),
            65..81 => (24, 3, 1),
            81.. => (24, 4, 1),
        }
    }

    /// Get the default prover options for a prover on CPU based on the amount of CPU memory.
    ///
    /// We use a soft heuristic based on our understanding of the memory usage in the GPU prover.
    #[must_use]
    pub fn cpu(cpu_ram_gb: usize) -> Self {
        let (_log2_shard_size, shard_batch_size, log2_divisor) = Self::get_memory_opts(cpu_ram_gb);

        let mut opts = DTProverOpts::default();
        opts.core_opts.shard_size = MAX_SHARD_SIZE;
        opts.core_opts.shard_batch_size = shard_batch_size;

        opts.core_opts.records_and_traces_channel_capacity = 1;
        opts.core_opts.trace_gen_workers = 1;

        let divisor = 1 << log2_divisor;
        opts.core_opts.split_opts.deferred /= divisor;
        opts.core_opts.split_opts.memory /= divisor;

        opts.recursion_opts.shard_batch_size = 2;
        opts.recursion_opts.records_and_traces_channel_capacity = 1;
        opts.recursion_opts.trace_gen_workers = 1;

        opts
    }

    /// Get the default prover options for a prover on GPU given the amount of CPU and GPU memory.
    #[must_use]
    pub fn gpu(cpu_ram_gb: usize, gpu_ram_gb: usize) -> Self {
        let mut opts = DTProverOpts::default();

        // Set the core options.
        if 24 <= gpu_ram_gb {
            let log2_shard_size = 24;
            opts.core_opts.shard_size = 1 << log2_shard_size;
            opts.core_opts.shard_batch_size = 1;

            let log2_deferred_threshold = 14;
            opts.core_opts.split_opts = SplitOpts::new(1 << log2_deferred_threshold);

            opts.core_opts.records_and_traces_channel_capacity = 4;
            opts.core_opts.trace_gen_workers = 4;

            if cpu_ram_gb <= 20 {
                opts.core_opts.records_and_traces_channel_capacity = 1;
                opts.core_opts.trace_gen_workers = 2;
            }
        } else {
            unreachable!("not enough gpu memory");
        }

        // Set the recursion options.
        opts.recursion_opts.shard_batch_size = 1;

        opts
    }
}
/// Options for the core prover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DTCoreOpts {
    /// The maximum number of RISC-V instructions per shard.
    ///
    /// This is NOT a height limit. The actual per-chip height is bounded by
    /// `sharding_threshold.height_threshold` (default 1<<22). The executor converts
    /// `shard_size` to `shard_size * 4` cycles internally as a hard-cap instruction count.
    pub shard_size: usize,
    /// The size of a batch of shards in terms of cycles.
    pub shard_batch_size: usize,
    /// The threshold that determines when to split the shard.
    pub sharding_threshold: ShardingThreshold,
    /// Options for splitting deferred events.
    pub split_opts: SplitOpts,
    /// The number of workers to use for generating traces.
    pub trace_gen_workers: usize,
    /// The capacity of the channel for checkpoints.
    pub checkpoints_channel_capacity: usize,
    /// The capacity of the channel for records and traces.
    pub records_and_traces_channel_capacity: usize,
}

impl Default for DTProverOpts {
    fn default() -> Self {
        Self {
            core_opts: DTCoreOpts::default(),
            recursion_opts: DTCoreOpts::recursion(),
            recursion_backend: None,
        }
    }
}

impl Default for DTCoreOpts {
    fn default() -> Self {
        let cpu_ram_gb = System::new_all().total_memory() / (1024 * 1024 * 1024);
        let (_default_log2_shard_size, default_shard_batch_size, default_log2_divisor) =
            DTProverOpts::get_memory_opts(cpu_ram_gb as usize);

        let element_threshold = if cpu_ram_gb >= 30 {
            SHARD_CELLS_THRESHOLD
        } else if cpu_ram_gb >= 20 {
            SHARD_CELLS_THRESHOLD_SMALL
        } else {
            // Allow running on machines with less than 20 GB RAM with a
            // reduced cell threshold.  The prover will be slower but still
            // functional.
            SHARD_CELLS_THRESHOLD_SMALL / 2
        };

        let mut opts = Self {
            shard_size: env::var("SHARD_SIZE")
                .map_or_else(|_| MAX_SHARD_SIZE, |s| s.parse::<usize>().unwrap_or(MAX_SHARD_SIZE)),
            shard_batch_size: env::var("SHARD_BATCH_SIZE").map_or_else(
                |_| default_shard_batch_size,
                |s| s.parse::<usize>().unwrap_or(default_shard_batch_size),
            ),
            sharding_threshold: ShardingThreshold {
                element_threshold,
                height_threshold: SHARD_HEIGHT_THRESHOLD,
            },
            split_opts: SplitOpts::new(MAX_DEFERRED_SPLIT_THRESHOLD),
            trace_gen_workers: env::var("TRACE_GEN_WORKERS").map_or_else(
                |_| DEFAULT_TRACE_GEN_WORKERS,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_TRACE_GEN_WORKERS),
            ),
            checkpoints_channel_capacity: env::var("CHECKPOINTS_CHANNEL_CAPACITY").map_or_else(
                |_| DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY),
            ),
            records_and_traces_channel_capacity: env::var("RECORDS_AND_TRACES_CHANNEL_CAPACITY")
                .map_or_else(
                    |_| DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY,
                    |s| s.parse::<usize>().unwrap_or(DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY),
                ),
        };

        let divisor = 1 << default_log2_divisor;
        opts.split_opts.deferred /= divisor;
        opts.split_opts.memory /= divisor;

        opts
    }
}

impl DTCoreOpts {
    /// Get the default options for the recursion prover.
    #[must_use]
    pub fn recursion() -> Self {
        let mut opts = Self::max();
        opts.shard_size = RECURSION_MAX_SHARD_SIZE;
        opts.shard_batch_size = 2;
        opts
    }

    /// Get the maximum options for the core prover.
    #[must_use]
    pub fn max() -> Self {
        let split_threshold = env::var("SPLIT_THRESHOLD")
            .map(|s| s.parse::<usize>().unwrap_or(MAX_DEFERRED_SPLIT_THRESHOLD))
            .unwrap_or(MAX_DEFERRED_SPLIT_THRESHOLD)
            .max(MAX_DEFERRED_SPLIT_THRESHOLD);

        let shard_size = env::var("SHARD_SIZE")
            .map_or_else(|_| MAX_SHARD_SIZE, |s| s.parse::<usize>().unwrap_or(MAX_SHARD_SIZE));

        Self {
            shard_size,
            shard_batch_size: env::var("SHARD_BATCH_SIZE").map_or_else(
                |_| MAX_SHARD_BATCH_SIZE,
                |s| s.parse::<usize>().unwrap_or(MAX_SHARD_BATCH_SIZE),
            ),
            sharding_threshold: ShardingThreshold {
                element_threshold: SHARD_CELLS_THRESHOLD,
                height_threshold: SHARD_HEIGHT_THRESHOLD,
            },
            split_opts: SplitOpts::new(split_threshold),
            trace_gen_workers: env::var("TRACE_GEN_WORKERS").map_or_else(
                |_| DEFAULT_TRACE_GEN_WORKERS,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_TRACE_GEN_WORKERS),
            ),
            checkpoints_channel_capacity: env::var("CHECKPOINTS_CHANNEL_CAPACITY").map_or_else(
                |_| DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY),
            ),
            records_and_traces_channel_capacity: env::var("RECORDS_AND_TRACES_CHANNEL_CAPACITY")
                .map_or_else(
                    |_| DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY,
                    |s| s.parse::<usize>().unwrap_or(DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY),
                ),
        }
    }
}

/// Options for splitting deferred events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitOpts {
    /// The threshold for combining the memory init/finalize events into the current shard in
    /// terms of cycles.
    pub combine_memory_threshold: usize,
    /// The threshold for default precompile events.
    pub deferred: usize,
    /// The threshold for keccak precompile events.
    pub keccak: usize,
    /// The threshold for sha extend precompile events.
    pub sha_extend: usize,
    /// The threshold for sha compress precompile events.
    pub sha_compress: usize,
    /// The threshold for memory events.
    pub memory: usize,
}

impl SplitOpts {
    /// Create a new [`SplitOpts`] with the given threshold.
    ///
    /// The constants here need to be chosen very carefully to prevent OOM. Consult @jtguibas on
    /// how to change them.
    #[must_use]
    pub fn new(deferred_split_threshold: usize) -> Self {
        Self {
            combine_memory_threshold: 1 << 17,
            deferred: deferred_split_threshold,
            keccak: 8 * deferred_split_threshold / 24,
            sha_extend: 32 * deferred_split_threshold / 48,
            sha_compress: 32 * deferred_split_threshold / 64,
            memory: 64 * deferred_split_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::*;

    #[test]
    fn test_opts() {
        let opts = DTProverOpts::cpu(8);
        println!("8: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(15);
        println!("15: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(16);
        println!("16: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(32);
        println!("32: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(36);
        println!("36: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(64);
        println!("64: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(128);
        println!("128: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(256);
        println!("256: {:?}", opts.core_opts);

        let opts = DTProverOpts::cpu(512);
        println!("512: {:?}", opts.core_opts);

        let opts = DTProverOpts::auto();
        println!("auto: {:?}", opts.core_opts);
    }
}
