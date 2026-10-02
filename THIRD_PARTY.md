# Third-party components

- **AWS Neuron NKI Library:** `benchmark-opt/nki-library/` contains adapters for [NKI Library](https://github.com/aws-neuron/nki-library), using revision `d2ad3a579498d1888aa8740af97fd325dd5f2740`. The dependency is obtained separately. Its Apache-2.0 license and notice are included in `third-party/`.
- **Hugging Face Transformers:** source comments in `benchmark-opt/CompilerGenBenchB2/` identify the Transformers v4.46.3 operations used to generate benchmark kernels with the Neuron compiler.
- **NKIBench:** source comments in `benchmark-opt/NKIBenchB2/` identify the original kernels and NKI API adaptations.
- **AutoComp:** `src/triton_nki_transpile/resources/opt-menus/` contains the comparison optimization menu.
- **NKI documentation:** API reference material is included under `src/triton_nki_transpile/resources/`.

Existing copyright notices and source attributions are retained in the corresponding files. Third-party components remain subject to their respective licenses.
