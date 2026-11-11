import pytest

from worldos.cli import main
from worldos.core.provenance import ValueKind
from worldos.data.catalogue import ingestion_allowed, load_catalogue
from worldos.data.observations import ObservationStore, ingest_csv
from worldos.validation.stats import gini


def test_catalogue_entries_complete():
    cat = load_catalogue()
    assert len(cat) >= 25
    assert all(e.license and e.url for e in cat)
    assert {"un_wpp", "wb_wdi", "era5", "openstreetmap", "fao_faostat", "ghsl"} <= {e.id for e in cat}


def test_licence_gate_and_csv_ingestion(tmp_path):
    cat = {e.id: e for e in load_catalogue()}
    csv = tmp_path / "pop.csv"
    csv.write_text("entity_id,variable,time,value,sd\nsyn,population,2020,100,5\nsyn,population,2021,101,\n")
    store = ObservationStore()
    assert ingest_csv(store, csv, cat["un_wpp"], unit="persons", sd_column="sd") == 2
    obs = store.query("syn", "population", ValueKind.OBSERVED)
    assert [o.value for o in obs] == [100, 101]
    assert obs[0].provenance.uncertainty.sd == 5 and obs[0].provenance.lineage[0].startswith("sha256:")
    with pytest.raises(PermissionError):
        ingest_csv(ObservationStore(), csv, cat["imf_data"], unit="USD")
    assert ingestion_allowed(cat["imf_data"], licence_confirmed_by="reviewer")
    with pytest.raises(ValueError):
        ingest_csv(ObservationStore(), csv, cat["un_wpp"], unit="furlongs")


def test_gini_known_values():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)
    assert gini([1, 3], [3, 1]) == pytest.approx(gini([1, 1, 1, 3]))


def test_cli_run_writes_outputs(tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text("[population]\nn_person_records = 2000\n[economy]\nn_firms = 40\n")
    assert main(["run", "--months", "13", "--config", str(cfg), "--out", str(tmp_path / "r")]) == 0
    for name in ("metrics.csv", "audit_log.json", "config.json", "dashboard.html"):
        assert (tmp_path / "r" / name).exists()
    html = (tmp_path / "r" / "dashboard.html").read_text()
    assert "not a forecast" in html
