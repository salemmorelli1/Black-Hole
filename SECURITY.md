# Security and numerical integrity

This repository does not require credentials or external services. Do not commit
tokens, private keys, local environment files, or unpublished research data.

Numerical-integrity issues should be reported when they could silently change:

- stopping behavior near `m_floor`;
- event compensators or terminal mark probabilities;
- Girsanov likelihood terms;
- score/Fisher derivative semantics;
- causal-filter conditioning;
- experiment seeds or artifact provenance.

Open a private security report through GitHub when the repository is published.
