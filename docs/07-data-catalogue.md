# 07 · Global Data Catalogue (Task 7)

The machine-readable catalogue is `worldos/data/catalogue.toml` (30 datasets), validated by
`worldos.data.catalogue.load_catalogue()` and listed with `python -m worldos catalogue`.

## 1. Policy

* **Licence status** is either `stated` (licence as published by the provider when catalogued;
  re-check on ingestion) or `verify` (terms are dataset-specific, recently changed, non-commercial,
  share-alike or otherwise need a human decision). Ingesting a `verify` dataset requires a named
  data steward (`licence_confirmed_by`); this is enforced in code.
* Licence statements are informational, not legal advice.
* **Alpha ingests none of these datasets.** It simulates a synthetic country. The ingestion code
  (CSV with schema, unit registry, finiteness checks, SHA-256 lineage and provenance) is tested on
  local files.

## 2. Coverage by domain

| Domain | Datasets |
|---|---|
| Population & demography | UN WPP, Human Mortality Database, WorldPop |
| Geography & settlements | Natural Earth, GADM (restricted), GHSL, OpenStreetMap |
| Economy, labour, distribution | World Bank WDI, IMF, OECD, Penn World Table, ILOSTAT, WID |
| Trade & supply chains | EXIOBASE, CEPII BACI |
| Agriculture & land | FAOSTAT, ESA WorldCover, SoilGrids, Global Forest Change |
| Climate & atmosphere | ERA5, HadCRUT5, NOAA GML CO₂, Global Carbon Budget, SSP database |
| Earth observation | NASA Earthdata |
| Energy | Ember, WRI Global Power Plant Database |
| Biodiversity | GBIF |
| Health & education | WHO GHO, UNESCO UIS |

## 3. Licence watch-list

| Dataset | Issue |
|---|---|
| GADM | Non-commercial; redistribution restricted → prefer Natural Earth / national open data |
| OpenStreetMap | ODbL share-alike applies to derived databases |
| WHO GHO | Non-commercial clause (CC BY-NC-SA 3.0 IGO) |
| GBIF | Licence varies per dataset (CC0 / BY / BY-NC) → filter on ingestion |
| EXIOBASE | Share-alike |
| IMF, SSP database, WID, UNESCO UIS | Custom terms → steward review |

## 4. Required metadata per observation

entity id · variable · time · value · kind (observed/estimated/inferred/simulated/forecast/assumed)
· source · unit (registry) · licence · reference date · geographic resolution · uncertainty
(sd or interval, or explicitly "unquantified") · lineage (content hash, upstream ids).

## 5. Next steps (M2)

1. Connectors for UN WPP (age-sex, ASFR, life tables) and World Bank WDI (macro targets) with cached
   raw files and checksums.
2. Mapping of WPP schedules to `DemographyConfig` (fit Siler/Hadwiger, record fit error as uncertainty).
3. First real-country calibration target set (docs/09 M3–M4).
