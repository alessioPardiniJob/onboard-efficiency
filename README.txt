=====================================
Project Setup, Execution & Reproducibility Guide
=====================================

This README describes, in precise technical detail, how to set up and reproduce the environment and tests for projects (e.g., EuroSAT, HyperView, etc.) that share a central `Utils/` package. It assumes each dataset/project has its own virtual environment under its directory. All instructions are given in a Unix-like context; Windows users should use WSL2 (Ubuntu or another Linux distro) for compatibility.

---------------------------
1. Prerequisites
---------------------------
- **Python 3.8+** installed on your system.
- **pip 21.0+**.
- **Git**.
- **WSL2** on Windows (strongly recommended) or native Linux/macOS.
- (Optional) **Make** if you wish to automate with a Makefile (examples provided).

Ensure that your `python3` and `pip` refer to the desired Python version. You can verify:
```bash
python3 --version
pip --version

---------------------------
2. Repository Layout
---------------------------
Assume you clone into `project-root/`. The top-level layout:


project-root/
├── Utils/                   # Shared utilities package
│   ├── __init__.py
│   ├── utils.py
│   ├── dp_utils.py
│   └── p_utils.py
│
├── EuroSAT/                 # Dataset/project 1
│   ├── code-Pardini/
│   │   ├── main.py
│   ├── code-DiPalma/
│   │   └── main.py
│   ├── configuration/       # JSONs and config files common to EuroSAT
│   ├── data/                # Data 
│   ├── result/              # Output directories
│   │   ├── result-Pardini/
│   │   └── result-DiPalma/
│   ├── venv/                # Virtual environment for EuroSAT
│   └── requirements.txt     # EuroSAT-specific dependencies 
│
├── HyperView/               # Dataset/project 2 (similar structure)
│   ├── code-Pardini/
│   │   ├── main.py
│   ├── code-DiPalma/
│   │   └── main.py
│   ├── configuration/       # JSONs and config files common to HyperView
│   ├── data/                # Data 
│   ├── result/              # Output directories
│   │   ├── result-Pardini/
│   │   └── result-DiPalma/
│   ├── venv/                # Virtual environment for EuroSAT
│   └── requirements.txt     # HyperView-specific dependencies
│
├── AnotherDataset/          # Additional dataset
│   ├── ...
│
├── setup.py                 # Defines installation for shared Utils package
└── README.txt               # (This file)

Each dataset (EuroSAT, HyperView, etc.) has its own venv/ under its folder.

The shared Utils/ package lives at the repository root.

setup.py at root installs only the Utils/ package in editable mode.

Each dataset’s requirements.txt must list dependencies needed to run its code and tests (e.g., pytest, ML libraries, etc.).

-==============================
Riproduzione Esperimenti
==============================

Per ogni dataset (es. EuroSAT, HyperView):

------------------------------
1. Setup dell'ambiente
------------------------------
make setup-EuroSAT

------------------------------
2. Esecuzione solo approccio ML (code-Pardini)
------------------------------
make test-EuroSAT-ML

------------------------------
3. Esecuzione solo approccio DL (code-DiPalma)
------------------------------
make test-EuroSAT-DL

------------------------------
4. Setup + Test ML
------------------------------
make all-EuroSAT-ML

------------------------------
5. Pulizia ambiente virtuale
------------------------------
make clean-EuroSAT







README: sezione d’uso di TMPDIR e no-cache
Aggiungi nel tuo README.md (o un file di documentazione) una sezione come la seguente per spiegare agli utenti come usarla:

markdown
Copy
Edit
## Impostazioni ambiente e gestione dello spazio durante l'installazione

Durante l’installazione delle dipendenze Python possono essere creati file temporanei o salvata cache di pip che occupano spazio su filesystem. Se il filesystem di default (`/tmp` o la root) è pieno o limitato, l’installazione potrebbe fallire con errori come “No space left on device”.

Per risolvere, il Makefile supporta due variabili opzionali:

- `TMPDIR`: percorso di una directory su un filesystem con spazio sufficiente. Se impostata, pip e altri tool useranno questa directory per i file temporanei.
- `NO_CACHE`: se impostata a `1`, pip installerà con l’opzione `--no-cache-dir`, evitando di salvare cache locale di pacchetti.

### Esempi di utilizzo

1. **Installazione su macchina con spazio sufficiente** (comportamento di default):
   ```bash
   make setup-EuroSAT
Vengono create/aggiornate le dipendenze nel venv predefinito EuroSAT/venv.

Installazione usando una directory temporanea su un disco più capiente:

bash
Copy
Edit
make setup-EuroSAT TMPDIR=/path/con/spazio/tmp
In questo modo pip e altri strumenti useranno /path/con/spazio/tmp per i file temporanei, evitando problemi di spazio in /tmp.

Installazione senza cache di pip:

bash
Copy
Edit
make setup-EuroSAT NO_CACHE=1
Pip installerà con --no-cache-dir, riducendo lo spazio usato dalla cache.

Combinare entrambe le opzioni:

bash
Copy
Edit
make setup-EuroSAT TMPDIR=/path/con/spazio/tmp NO_CACHE=1
Massima sicurezza contro errori di spazio: i file temporanei vanno nella cartella specificata, pip non salva cache locale.


così funziona:

ui questi due comandi:

Crea una directory temporanea dedicata su disk4 (buona pratica):

Bash

mkdir -p /disk4/apardini-gdpalma-paper/tmp
Lancia make usando la variabile TMPDIR per puntare a quella directory:

Bash

make setup-EuroSAT TMPDIR=/disk4/apardini-gdpalma-paper/tmp



4. Pulizia dell'Ambiente
Per rimuovere l'ambiente virtuale e gli artefatti di setup (come il file .egg-info e l'eventuale TMPDIR utilizzata) per un dataset specifico:

make clean-<NomeDataset>

Esempi:

make clean-EuroSAT
make clean-HyperView

Importante: Se hai specificato TMPDIR durante il setup, assicurati di specificarla anche durante la pulizia per rimuovere quella directory temporanea.

make clean-EuroSAT TMPDIR=/disk4/apardini-gdpalma-paper/tmp

Questa operazione NON rimuove i file o le directory dei risultati (modelli, submission CSV, ecc.).

make clean-EuroSAT TMPDIR=/disk4/apardini-gdpalma-paper/tmp
