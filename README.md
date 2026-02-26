## Project structure
Fetch yearly CPI basket weights from SCB and export:
- CSV wide table by category and year
- Interactive stacked-share HTML chart

  
- `scripts/cpi_weights_from_scb_api.py`
- `data/` (generated CSV)
- `figures/` (generated HTML)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python3 scripts/cpi_weights_from_scb_api.py
```

Default outputs:
- `data/scb_cpi_weights_major_wide.csv`
- `figures/scb_weights_share_stacked_by_year_major_categories.html`

## Notes
- Uses SCB API endpoint `KPI2020COICOP2M`.
- Excludes category `00` (total basket) from share chart.
