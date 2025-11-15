# Toolkit
This is a code repository that stores commonly used tools.

# SMPL High-Resolution Mesh Subdivision Toolkit (meshsubdivide)

This repository provides tools for subdividing the SMPL mesh and generating a high-resolution SMPL model with fully updated parametric components.

## Features
- Subdivide the original 6890-vertex SMPL mesh  
- Track vertex origin pairs for consistent parameter interpolation  
- Recompute:
  - `J_regressor`
  - `J_regressor_extra`
  - `shapedirs`
  - `posedirs`
  - `weights`
- Export a complete high-resolution SMPL model
- Provide SMPL+D mesh fitting tools for validation

## Files
- **meshsubdivide.py** — Mesh subdivision and high-resolution parameter generation  
- **model_test.py** — Testing and visualization of the subdivided model  
- **meshfitting.py** — SMPL+D fitting to target meshes  
- **util/config.py** — Configuration of input/output paths  

---

_Last updated: **2025-02-15**_
