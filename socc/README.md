# SOC to C-lang port

Python is undeniable good prototype platform, SOCC will be a port in c-language.  
The reason is that C-compiler/framework makes use of `SSE1/2` instruction sets which provide more performance.  

Usage:  
`make clean && make all`  *output in build folder*

# Project file structure
```
  socc/
    ├── ffi.py               # loads libsoc.so, defines ctypes signatures
    ├── constants.py         # Python-side constants mirroring types.h
    ├── shape.py             # Shape class wrapping SocShape
    ├── operators.py         # operator functions bound to C
    ├── metrics.py           # eta, y_rx, energy_audit
    ├── generators.py        # circle, polygon, star, curve_to_coeffs
    └── pipeline.py          # Pipeline class (Python-side composition)
```
