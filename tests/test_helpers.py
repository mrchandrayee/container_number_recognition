import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.infer_container import normalize_container_code, parse_crop_rect, parse_polygon_points

def test_normalize():
    assert normalize_container_code("mscu 123456-7") == "MSCU1234567"

def test_crop():
    assert parse_crop_rect("1,2,30,40") == (1, 2, 30, 40)

def test_polygon():
    assert parse_polygon_points("0,0,10,0,10,10") == ((0, 0), (10, 0), (10, 10))
