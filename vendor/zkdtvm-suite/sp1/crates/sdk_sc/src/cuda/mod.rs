// //! # zkDTVM CUDA Prover
// //!
// //! A prover that uses the CUDA to execute and prove programs.

// pub mod builder;
// pub mod prove;

// use anyhow::Result;
// use prove::CudaProveBuilder;
// use dt_core_executor::DTContextBuilder;
// use dt_core_machine::io::DTStdin;
// use dt_cuda::{MoongateServer, DTCudaProver};
// use dt_prover::{components::SCCpuProverComponents, SCDTProver};

// use crate::{
//     cpu::execute::CpuExecuteBuilder, install::try_install_circuit_artifacts, Prover,
//     DTProvingKey, DTVerifyingKey, DTProof, DTProofMode, DTProofWithPublicValues,
// };

// /// A prover that uses the CPU for execution and the CUDA for proving.
// pub struct CudaProver {
//     pub(crate) cpu_prover: SCDTProver<SCCpuProverComponents>,
//     pub(crate) cuda_prover: DTCudaProver,
// }

// impl CudaProver {
//     /// Creates a new [`CudaProver`].
//     pub fn new(prover: SCDTProver, moongate_server: MoongateServer) -> Self {
//         let cuda_prover = DTCudaProver::new(moongate_server);
//         Self {
//             cpu_prover: prover,
//             cuda_prover: cuda_prover.expect("Failed to initialize CUDA prover"),
//         }
//     }

//     /// Creates a new [`CpuExecuteBuilder`] for simulating the execution of a program on the CPU.
//     ///
//     /// # Details
//     /// The builder is used for both the [`crate::cpu::CpuProver`] and [`crate::CudaProver`]
// client     /// types.
//     ///
//     /// # Example
//     /// ```rust,no_run
//     /// use dt_sdk::{include_elf, Prover, ProverClient, DTStdin};
//     ///
//     /// let elf = &[1, 2, 3];
//     /// let stdin = DTStdin::new();
//     ///
//     /// let client = ProverClient::builder().cuda().build();
//     /// let (public_values, execution_report) = client.execute(elf, &stdin).run().unwrap();
//     /// ```
//     pub fn execute<'a>(&'a self, elf: &'a [u8], stdin: &DTStdin) -> CpuExecuteBuilder<'a> {
//         CpuExecuteBuilder {
//             prover: &self.cpu_prover,
//             elf,
//             stdin: stdin.clone(),
//             context_builder: DTContextBuilder::default(),
//         }
//     }

//     /// Creates a new [`CudaProveBuilder`] for proving a program on the CUDA.
//     ///
//     /// # Details
//     /// The builder is used for only the [`crate::CudaProver`] client type.
//     ///
//     /// # Example
//     /// ```rust,no_run
//     /// use dt_sdk::{include_elf, Prover, ProverClient, DTStdin};
//     ///
//     /// let elf = &[1, 2, 3];
//     /// let stdin = DTStdin::new();
//     ///
//     /// let client = ProverClient::builder().cuda().build();
//     /// let (pk, vk) = client.setup(elf);
//     /// let proof = client.prove(&pk, &stdin).run().unwrap();
//     /// ```
//     pub fn prove<'a>(
//         &'a self,
//         pk: &'a DTProvingKey,
//         stdin: &'a DTStdin,
//     ) -> CudaProveBuilder<'a> {
//         CudaProveBuilder { prover: self, mode: DTProofMode::Core, pk, stdin: stdin.clone() }
//     }

//     /// Proves the given program on the given input in the given proof mode.
//     ///
//     /// Returns the cycle count in addition to the proof.
//     pub fn prove_with_cycles(
//         &self,
//         pk: &DTProvingKey,
//         stdin: &DTStdin,
//         kind: DTProofMode,
//     ) -> Result<(DTProofWithPublicValues, u64)> {
//         // Generate the core proof.
//         let proof = self.cuda_prover.prove_core_stateless(pk, stdin)?;
//         // TODO: Return the prover gas
//         let cycles = proof.cycles;
//         if kind == DTProofMode::Core {
//             let proof_with_pv = DTProofWithPublicValues::new(
//                 DTProof::Core(proof.proof.0),
//                 proof.public_values,
//                 self.version().to_string(),
//             );
//             return Ok((proof_with_pv, cycles));
//         }

//         // Generate the compressed proof.
//         let deferred_proofs =
//             stdin.proofs.iter().map(|(reduce_proof, _)| reduce_proof.clone()).collect();
//         let public_values = proof.public_values.clone();
//         let reduce_proof = self.cuda_prover.compress(&pk.vk, proof, deferred_proofs)?;
//         if kind == DTProofMode::Compressed {
//             let proof_with_pv = DTProofWithPublicValues::new(
//                 DTProof::Compressed(Box::new(reduce_proof)),
//                 public_values,
//                 self.version().to_string(),
//             );
//             return Ok((proof_with_pv, cycles));
//         }

//         // Generate the shrink proof.
//         let compress_proof = self.cuda_prover.shrink(reduce_proof)?;

//         // Genenerate the wrap proof.
//         let outer_proof = self.cuda_prover.wrap_bn254(compress_proof)?;

//         if kind == DTProofMode::Plonk {
//             let plonk_bn254_artifacts = if dt_prover::build::dt_dev_mode() {
//                 dt_prover::build::try_build_plonk_bn254_artifacts_dev(
//                     &outer_proof.vk,
//                     &outer_proof.proof,
//                 )
//             } else {
//                 try_install_circuit_artifacts("plonk")
//             };
//             let proof = self.cpu_prover.wrap_plonk_bn254(outer_proof, &plonk_bn254_artifacts);
//             let proof_with_pv = DTProofWithPublicValues::new(
//                 DTProof::Plonk(proof),
//                 public_values,
//                 self.version().to_string(),
//             );
//             return Ok((proof_with_pv, cycles));
//         } else if kind == DTProofMode::Groth16 {
//             let groth16_bn254_artifacts = if dt_prover::build::dt_dev_mode() {
//                 dt_prover::build::try_build_groth16_bn254_artifacts_dev(
//                     &outer_proof.vk,
//                     &outer_proof.proof,
//                 )
//             } else {
//                 try_install_circuit_artifacts("groth16")
//             };

//             let proof = self.cpu_prover.wrap_groth16_bn254(outer_proof,
// &groth16_bn254_artifacts);             let proof_with_pv = DTProofWithPublicValues::new(
//                 DTProof::Groth16(proof),
//                 public_values,
//                 self.version().to_string(),
//             );
//             return Ok((proof_with_pv, cycles));
//         }

//         unreachable!()
//     }
// }

// impl Prover<SCCpuProverComponents> for CudaProver {
//     fn setup(&self, elf: &[u8]) -> (DTProvingKey, DTVerifyingKey) {
//         let (pk, vk) = self.cuda_prover.setup(elf).unwrap();
//         (pk, vk)
//     }

//     fn inner(&self) -> &SCDTProver<SCCpuProverComponents> {
//         &self.cpu_prover
//     }

//     fn prove(
//         &self,
//         pk: &DTProvingKey,
//         stdin: &DTStdin,
//         kind: DTProofMode,
//     ) -> Result<DTProofWithPublicValues> {
//         self.prove_with_cycles(pk, stdin, kind).map(|(p, _)| p)
//     }
// }

// impl Default for CudaProver {
//     fn default() -> Self {
//         Self::new(SCDTProver::new(), MoongateServer::default())
//     }
// }
