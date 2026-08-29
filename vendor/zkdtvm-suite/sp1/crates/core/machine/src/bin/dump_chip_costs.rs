use dt_core_machine::riscv::RiscvAir;
use dt_primitives::SCField;
use dt_stark::air::MachineAir;
use p3_air::BaseAir;

fn main() {
    let (chips, costs) = RiscvAir::<SCField>::get_chips_and_costs();

    println!("{{");
    let mut entries: Vec<_> = costs.iter().collect();
    entries.sort_by_key(|(name, _)| (*name).clone());
    for (i, (name, cost)) in entries.iter().enumerate() {
        let comma = if i < entries.len() - 1 { "," } else { "" };
        println!("  \"{name}\": {cost}{comma}");
    }
    println!("}}");

    println!("\n--- Detailed breakdown ---");
    for chip in &chips {
        let name = MachineAir::<SCField>::name(chip);
        let preprocessed = MachineAir::<SCField>::preprocessed_width(chip);
        let main = <_ as BaseAir<SCField>>::width(chip);
        let perm = chip.permutation_width() * 4;
        let total = preprocessed + main + perm;
        println!(
            "{name:30} preprocessed={preprocessed:4} main={main:4} perm={perm:4} total={total:5}"
        );
    }
}
