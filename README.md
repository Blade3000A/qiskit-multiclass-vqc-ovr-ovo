# qiskit-multiclass-vqc-ovr-ovo
Extends Qiskit's VQC to improve multiclass classification on the Iris dataset. Instead of relying on the built-in multiclass handler, three binary classifiers are trained using a balanced OvR strategy. A second OvO decision layer resolves ambiguous predictions, improving accuracy over the single-model baseline.
