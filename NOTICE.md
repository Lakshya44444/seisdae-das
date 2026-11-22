# Credits and licences

- **seisDAE.** The network architecture, the training code (used unchanged through `scripts/train_seisdae.py`) and the released model in `models/gr_mixed_stft.*` come from Janis Heuel's [seisDAE](https://github.com/JanisHe/seisDAE/tree/26c9ba0fd70c532f0fb1a612d816cd32475b5036), licensed under GPL-3.0. `LICENSE` is that GPL-3.0 text, and this repository's code is released under the same licence.
- **DAS data.** Ridgecrest DAS dataset of the Southern California Earthquake Data Center (`s3://scedc-pds/Ridgecrest_DAS`). Cite Li, Z., et al. (2021), *AGU Advances*, 2, e2021AV000395, https://doi.org/10.1029/2021AV000395.
- **Earthquake catalogue.** SCEDC FDSN event service, accessed through ObsPy.
- **Earlier attempt.** Third-party data used by the earlier attempt in `archive/earlier_attempt/` are credited in `archive/earlier_attempt/docs/`.
