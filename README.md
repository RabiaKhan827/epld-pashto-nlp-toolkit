# epld-pashto-nlp-toolkit
Official Python NLP toolkit, morphology-aware # EPLD Pashto–English NLP Toolkit (`epld-pashto-nlp-toolkit`)

Official companion Python library, morphological analyzer, and cross-lingual sentiment benchmark for the peer-reviewed open-access **English-Pashto Language Dataset (EPLD)**:

> **Khan, Rabia**, Khan, Huzaifa Saleem, & Ijaz, Shireen (2025). *English-Pashto Language Dataset (EPLD)*. **Mendeley Data**, V1. DOI: `10.17632/vmgv4s6vrn.1`

## Author & Academic Context
- **Lead Author & Maintainer:** **Rabia Khan**
- **Current Appointment:** Lecturer, Department of Information Technology, University of Haripur, Khyber Pakhtunkhwa, Pakistan
- **Academic Credentials:**
  - **M.Sc. in Data Science**, Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology (PAF-IAST), Haripur
  - **B.Sc. in Computer Engineering**, COMSATS University Islamabad, Abbottabad Campus
- **ORCID:** `0009-0007-5575-1564`

## Core Modules & Features
1. **Orthographic Normalization (`epld_nlp.normalizer`):** Standardizes Perso-Arabic character variants (`ك -> ک`, `گ -> ګ`, `ي -> ی`) and Pashto-specific retroflex/affricate phonemes (`ټ، ډ، ړ، ږ، ښ، ګ، ڼ، څ، ځ`).
2. **Morphology-Aware Tokenization (`epld_nlp.tokenizer`):** Rule-based stemmer and affix analyzer supporting Pashto split-ergative pronouns (`زه -> ما`, `ته -> تا`), nominal plural suffixes (`-ونه`, `-انې`, `-ګانې`), and directional verbal prefixes (`را-`, `در-`, `ور-`, `نه-`).
3. **EPLD Corpus Loader (`epld_nlp.corpus`):** Structured interface for the 4 EPLD modules (`Alphabets`, `Counting 0-100`, `Pronouns`, and `Parallel Sentences`).
4. **Cross-Lingual Sentiment Pipeline (`epld_nlp.sentiment`):** Lexical and morphology-informed sentiment classifier achieving `1.0000` Macro-F1 on the sample benchmark split.

## Quickstart
```bash
pip install -e .
python -m unittest discover -s tests -v
```

## Citation
```bibtex
@dataset{khan2025epld,
  author    = {Khan, Rabia and Khan, Huzaifa Saleem and Ijaz, Shireen},
  title     = {{English-Pashto Language Dataset (EPLD)}},
  year      = {2025},
  publisher = {Mendeley Data},
  version   = {V1},
  doi       = {10.17632/vmgv4s6vrn.1}
}
```
