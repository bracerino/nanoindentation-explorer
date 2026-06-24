# Nanoindentation Explorer
Interactive app for exploring and comparing nanoindentation (nanoDMA) depth profiles - storage modulus, hardness and contact depth

Upload one or many `*_DYN.txt` files exported from Hysitron/Bruker instruments and:
  - plot **storage modulus**, **hardness** or **contact depth** vs. depth,
  - overlay individual indents or compute **mean / median ± std / SEM** per grain,
  - define **custom groups** of files (e.g. different samples or fluences) and
    compare their averaged curves,
  - filter by sample / fluence / grain, and export the plotted curves to CSV.

File names like `s40_f46_g15_000_DYN.txt` are decoded automatically into
**sample / fluence / grain / measurement**.

Use the app online at: 
or compile it locally:
- 
  ## Run
  ```bash
  git copy 
  pip install -r requirements.txt
  streamlit run app.py
