use p3_field::Field;

pub use dt_primitives::consts::{
    bytes_to_words_le, bytes_to_words_le_vec, num_to_comma_separated, words_to_bytes_le,
    words_to_bytes_le_vec,
};

use crate::shape::{CHIP_LOG_HEIGHT_THRESHOLD, NUM_SKIP_ROUNDS};

pub const fn indices_arr<const N: usize>() -> [usize; N] {
    let mut indices_arr = [0; N];
    let mut i = 0;
    while i < N {
        indices_arr[i] = i;
        i += 1;
    }
    indices_arr
}

pub fn pad_to_power_of_two<const N: usize, T: Clone + Default>(values: &mut Vec<T>) {
    debug_assert!(values.len().is_multiple_of(N));
    let mut n_real_rows = values.len() / N;
    if n_real_rows < 16 {
        n_real_rows = 16;
    }
    values.resize(n_real_rows.next_power_of_two() * N, T::default());
}

pub fn padded_rows_threshold(padded_nb_rows: usize) -> usize {
    let log2 = log2_strict_usize(padded_nb_rows);
    if log2 < CHIP_LOG_HEIGHT_THRESHOLD {
        1 << (log2.div_ceil(NUM_SKIP_ROUNDS) * NUM_SKIP_ROUNDS)
    } else {
        padded_nb_rows
    }
}

pub fn pad_rows_fixed<R: Clone>(
    rows: &mut Vec<R>,
    row_fn: impl Fn() -> R,
    size_log2: Option<usize>,
) {
    let nb_rows = rows.len();
    let dummy_row = row_fn();
    let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));
    rows.resize(padded_nb_rows, dummy_row);
}

pub fn next_power_of_two(n: usize, fixed_power: Option<usize>) -> usize {
    match fixed_power {
        Some(power) => {
            let padded_nb_rows = 1 << power;
            if n > padded_nb_rows {
                let mut fallback = n.next_power_of_two();
                if fallback < 16 {
                    fallback = 16;
                }
                return fallback;
            }
            padded_nb_rows
        }
        None => {
            let mut padded_nb_rows = n.next_power_of_two();
            if padded_nb_rows < 16 {
                padded_nb_rows = 16;
            }
            padded_nb_rows
        }
    }
}

pub fn chunk_vec<T>(mut vec: Vec<T>, chunk_size: usize) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    while !vec.is_empty() {
        let current_chunk_size = std::cmp::min(chunk_size, vec.len());
        let current_chunk = vec.drain(..current_chunk_size).collect::<Vec<T>>();
        result.push(current_chunk);
    }
    result
}

#[inline]
pub fn log2_strict_usize(n: usize) -> usize {
    let res = n.trailing_zeros();
    assert_eq!(n.wrapping_shr(res), 1, "Not a power of two: {n}");
    res as usize
}

pub fn par_for_each_row<P, F>(vec: &mut [F], num_elements_per_event: usize, processor: P)
where
    F: Send,
    P: Fn(usize, &mut [F]) + Send + Sync,
{
    use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
    assert!(vec.len().is_multiple_of(num_elements_per_event));
    let len = vec.len() / num_elements_per_event;
    let cpus = num_cpus::get();
    let ceil_div = len.div_ceil(cpus);
    let chunk_size = std::cmp::max(ceil_div, cpus);
    vec.chunks_mut(chunk_size * num_elements_per_event).enumerate().par_bridge().for_each(
        |(i, chunk)| {
            chunk.chunks_mut(num_elements_per_event).enumerate().for_each(|(j, row)| {
                assert!(row.len() == num_elements_per_event);
                processor(i * chunk_size + j, row);
            });
        },
    );
}

pub fn setup_logger() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

pub fn dt_debug_mode() -> bool {
    let value = std::env::var("DT_DEBUG").unwrap_or_else(|_| "false".to_string());
    value == "1" || value.to_lowercase() == "true"
}

pub fn zeroed_f_vec<F: Field>(len: usize) -> Vec<F> {
    debug_assert!(std::mem::size_of::<F>() == 4);
    let vec = vec![0u32; len];
    unsafe { std::mem::transmute::<Vec<u32>, Vec<F>>(vec) }
}

pub fn run_test_machine<SC, A, AE>(
    _records: Vec<A::Record>,
    _machine: dt_stark::SCStarkMachine<SC, A, AE>,
    _pk: dt_stark::sumcheck::keys::SCStarkProvingKey<SC>,
    _vk: dt_stark::sumcheck::keys::SCStarkVerifyingKey<SC>,
) where
    SC: dt_stark::sumcheck::config::SCStarkGenericConfig,
    A: dt_stark::air::MachineAir<SC::Val>,
    AE: dt_stark::air::MachineAir<SC::Val>,
{
    unimplemented!("run_test_machine is prover-only and not available in verifier build")
}
