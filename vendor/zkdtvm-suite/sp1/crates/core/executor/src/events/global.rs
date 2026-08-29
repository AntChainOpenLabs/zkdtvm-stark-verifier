use serde::{Deserialize, Serialize};

/// Stable identifiers for the ordered batches that feed the Global chip.
///
/// The numeric order is proof-adjacent host metadata: Global materialization must visit these
/// sources in this exact order, independently of parallel completion or map iteration order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(u8)]
pub enum GlobalSourceId {
    /// Forwarded syscalls emitted directly by the core syscall instruction AIR.
    CoreSyscall = 0,
    /// Deferred/precompile syscall dispatches.
    DeferredSyscall = 1,
    /// Global-memory initialization boundaries.
    MemoryInitialize = 2,
    /// Global-memory finalization boundaries.
    MemoryFinalize = 3,
    /// Initial and final endpoints of local-memory lifetimes.
    MemoryLocal = 4,
    /// SHA-256 extend controller completions.
    ShaExtendController = 5,
    /// SHA-256 compress controller completions.
    ShaCompressController = 6,
    /// Keccak controller completions.
    KeccakController = 7,
}

impl GlobalSourceId {
    /// Ordered production schedule for Global endpoint materialization.
    pub const ALL: [Self; 8] = [
        Self::CoreSyscall,
        Self::DeferredSyscall,
        Self::MemoryInitialize,
        Self::MemoryFinalize,
        Self::MemoryLocal,
        Self::ShaExtendController,
        Self::ShaCompressController,
        Self::KeccakController,
    ];

    /// Stable producer ordinal.
    #[inline]
    #[must_use]
    pub const fn ordinal(self) -> u16 {
        self as u16
    }

    /// Stable diagnostic label.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CoreSyscall => "core_syscall",
            Self::DeferredSyscall => "deferred_syscall",
            Self::MemoryInitialize => "memory_initialize",
            Self::MemoryFinalize => "memory_finalize",
            Self::MemoryLocal => "memory_local",
            Self::ShaExtendController => "sha_extend_controller",
            Self::ShaCompressController => "sha_compress_controller",
            Self::KeccakController => "keccak_controller",
        }
    }
}

/// Global Interaction Event.
///
/// This event is emitted for all interactions that are sent or received across different shards.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct GlobalInteractionEvent {
    /// The message.
    pub message: [u32; 7],
    /// Whether the interaction is received or sent.
    pub is_receive: bool,
    /// The kind of the interaction event.
    pub kind: u8,
}
