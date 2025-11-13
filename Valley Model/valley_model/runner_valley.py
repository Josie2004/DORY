
# runner_valley.py
# Convenience runner: python runner_valley.py params_valley.xml
import sys
from valley_model.xmlio_valley import parse_valley_xml
from valley_model.model import build_valley_model

if __name__ == "__main__":
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "params_valley.xml"
    params = parse_valley_xml(xml_path)
    E, V, H = build_valley_model(params, report=True)
