use dt_stark::{air::FullAirBuilder, InteractionKind};
use p3_field::AbstractField;

mod sealed {
    pub trait Sealed {}
}

/// A structurally fixed kind for an AIR sender on the Global bus.
///
/// Implementations are sealed so production senders cannot smuggle a witness-selected `kind`
/// through host metadata.
pub trait GlobalKind: sealed::Sealed {
    /// Constant payload kind embedded in the Global interaction expression.
    const KIND: InteractionKind;
}

/// Memory transition endpoint.
pub enum MemoryGlobalKind {}

impl sealed::Sealed for MemoryGlobalKind {}

impl GlobalKind for MemoryGlobalKind {
    const KIND: InteractionKind = InteractionKind::Memory;
}

/// Syscall dispatch/completion endpoint.
pub enum SyscallGlobalKind {}

impl sealed::Sealed for SyscallGlobalKind {}

impl GlobalKind for SyscallGlobalKind {
    const KIND: InteractionKind = InteractionKind::Syscall;
}

/// Build the exact ten-value Global lookup denominator with a sealed constant kind.
pub fn global_lookup_denominator<AB, K>(
    builder: &AB,
    message: [AB::VarMaybeExt; 7],
    is_send: AB::VarMaybeExt,
    is_receive: AB::VarMaybeExt,
) -> AB::VarExt
where
    AB: FullAirBuilder,
    K: GlobalKind,
{
    let global_bus_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Global as usize));
    let endpoint_kind = AB::VarMaybeExt::from(AB::F::from_canonical_u8(K::KIND as u8));
    let [m0, m1, m2, m3, m4, m5, m6] = message;

    builder.lookup_denominator(
        global_bus_kind,
        [m0, m1, m2, m3, m4, m5, m6, is_send, is_receive, endpoint_kind],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealed_kinds_match_global_payload_contract() {
        assert_eq!(MemoryGlobalKind::KIND, InteractionKind::Memory);
        assert_eq!(SyscallGlobalKind::KIND, InteractionKind::Syscall);
    }
}
