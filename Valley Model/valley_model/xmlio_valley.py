
# xmlio_valley.py
# Minimal XML parser for the <valley_model> block.
import xml.etree.ElementTree as ET

def parse_bool(s: str) -> bool:
    return str(s).strip().lower() in {'1','true','yes','on'}

def parse_valley_xml(path: str) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    vm = root.find('valley_model')
    if vm is None:
        raise ValueError("No <valley_model> block found in XML.")

    params = {}
    params['enabled'] = parse_bool(vm.get('enabled', 'true'))
    params['spin']    = parse_bool(vm.get('spin', 'true'))

    # ordering
    ordering_tag = vm.findtext('ordering', '+x,-x,+y,-y,+z,-z')
    params['ordering'] = [s.strip() for s in ordering_tag.split(',') if s.strip()]

    # delta_c and delta
    def _num(text, default=0.0):
        if text is None:
            return default
        try:
            return float(text.strip())
        except Exception:
            return default

    delta_c_text = vm.findtext('delta_c', '2.5')
    delta_text   = vm.findtext('delta', '0.0')
    params['delta_c'] = _num(delta_c_text, 2.5)
    params['delta']   = _num(delta_text, 0.0)

    # valley_shifts (optional)
    shifts_tag = vm.find('valley_shifts')
    shifts = {}
    if shifts_tag is not None:
        for key in ['Ex','Ex_bar','Ey','Ey_bar','Ez','Ez_bar']:
            shifts[key] = _num(shifts_tag.findtext(key), 0.0)
    params['valley_shifts'] = shifts

    # ... inside parse_valley_xml(...)
    baseline_text = vm.findtext('baseline_meV', None)

    def _num(text, default=0.0):
        if text is None: return default
        try: return float(text.strip())
        except Exception: return default

    # existing reads...
    params['baseline_meV'] = _num(baseline_text, 0.0)

    return params


