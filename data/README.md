# Real-data extracts and attributions

Small, filtered extracts of public datasets, committed so that every test, calibration and report
runs offline and is reproducible. Each folder has a `manifest.json` with the source URL, retrieval
time (UTC), filters and a SHA-256 checksum per file (`python -m worldos data verify`). Refresh with
`python -m worldos data fetch [--countries IND,USA,...]`.

Countries included: India, United States, Germany, Japan, Brazil, Nigeria.

| Folder | Source | Licence | Attribution |
|---|---|---|---|
| `real/wpp2024/` | UN World Population Prospects 2024 (bulk CSV) | CC BY 3.0 IGO | United Nations, Department of Economic and Social Affairs, Population Division (2024). World Population Prospects 2024. |
| `real/wdi/` | World Bank World Development Indicators (API v2) | CC BY 4.0 | World Bank, World Development Indicators. |
| `real/naturalearth/` | Natural Earth 10m admin-0, admin-1, populated places (simplified) | Public domain | Made with Natural Earth. |
| `real/kontur/` | Kontur Population 2023-11-01 (H3 r8), aggregated to each country's model H3 resolution | CC BY 4.0 | Kontur Population Dataset, Kontur Inc., via the Humanitarian Data Exchange. |
| `real/climate/emissions_owid_gcb.csv` | Global Carbon Budget via Our World in Data CO2 dataset | CC BY 4.0 | Friedlingstein et al., Global Carbon Budget; Our World in Data. |
| `real/climate/co2_mauna_loa_annual.csv` | NOAA GML Mauna Loa annual mean CO2 | US Government work | Dr. Xin Lan, NOAA/GML, and Dr. Ralph Keeling, Scripps Institution of Oceanography. |
| `real/climate/hadcrut5_global_annual.csv` | HadCRUT5 analysis, global annual | Open Government Licence v3 | Morice et al. (2021), Met Office Hadley Centre / UEA CRU. |
| `real/climate/ar6_erf_1750_2019.csv` | IPCC AR6 WG1 Chapter 7 effective radiative forcing | CC BY 4.0 (data), MIT (code) | Smith, C. et al. (2021), IPCC AR6 WG1 Chapter 7 supplementary data. |

Modifications: rows filtered to the listed countries and years; columns subset; Natural Earth
geometries simplified (0.02°). Values are otherwise unchanged.

`calibration/` holds model parameters calibrated by WORLD-OS (not source data).
