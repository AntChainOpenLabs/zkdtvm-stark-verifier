use std::{cell::RefCell, collections::BTreeMap};
use web_time::Instant;

use p3_commit::{MmcsCommitObserver, MmcsCommitPhase};

thread_local! {
    static PROFILE_MS: RefCell<BTreeMap<&'static str, u128>> = RefCell::new(BTreeMap::new());
}

pub fn reset() {
    PROFILE_MS.with(|profile| profile.borrow_mut().clear());
}

pub fn take() -> BTreeMap<String, u128> {
    PROFILE_MS.with(|profile| {
        let mut profile = profile.borrow_mut();
        std::mem::take(&mut *profile)
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect()
    })
}

pub fn add_ms(label: &'static str, ms: u128) {
    PROFILE_MS.with(|profile| {
        *profile.borrow_mut().entry(label).or_default() += ms;
    });
}

pub fn time<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let output = f();
    add_ms(label, start.elapsed().as_millis());
    output
}

pub struct MmcsCommitProfiler {
    leaf_hash_start: Option<Instant>,
    tree_build_start: Option<Instant>,
}

impl MmcsCommitProfiler {
    pub const fn new() -> Self {
        Self {
            leaf_hash_start: None,
            tree_build_start: None,
        }
    }
}

impl Default for MmcsCommitProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MmcsCommitObserver for MmcsCommitProfiler {
    fn start(&mut self, phase: MmcsCommitPhase) {
        match phase {
            MmcsCommitPhase::LeafHash => self.leaf_hash_start = Some(Instant::now()),
            MmcsCommitPhase::TreeBuild => self.tree_build_start = Some(Instant::now()),
        }
    }

    fn end(&mut self, phase: MmcsCommitPhase) {
        let (slot, label) = match phase {
            MmcsCommitPhase::LeafHash => (&mut self.leaf_hash_start, "commit.mmcs_leaf_hash_ms"),
            MmcsCommitPhase::TreeBuild => (&mut self.tree_build_start, "commit.mmcs_tree_build_ms"),
        };
        if let Some(start) = slot.take() {
            add_ms(label, start.elapsed().as_millis());
        }
    }
}
