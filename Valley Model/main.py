
import sys
from valley_model.model import build_valley_model
from valley_model.xmlio_valley import parse_valley_xml

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "valley"
    xml_path = sys.argv[2] if len(sys.argv) > 2 else "params_valley.xml"

    if mode == "valley":
        params = parse_valley_xml(xml_path)
        E, V, H = build_valley_model(params, report=True)
        # Minimal pretty print
        print("\n=== Results ===")
        for i, e in enumerate(E):
            print(f"Level {i:>2}: {e:.6f} meV")
    else:
        raise SystemExit("Unknown mode. Try: python main.py valley params_valley.xml")

if __name__ == "__main__":
    main()
