# DopantModeling  

**DopantModeling** was developed to study how doping of conjugated polymers affects their X-ray optical properties and the resulting polarized resonant soft X-ray scattering (P-RSoXS). It is designed for use alongside **NRSS** and **CyRSoXS** to simulate P-RSoXS of models generated by **DopantModeling**.  

These models are voxel-based, with each voxel defined by:  
1. The number of individual materials it contains.  
2. The volume fraction of each material.  
3. The orientation of the material within the voxel.  
4. A parameter denoting the magnitude or extent of the orientation.  

---

## Model Parameters  

Key parameters for defining models include:  
- Model dimensions (in nm).  
- Model pitch (in nm), determining voxel size and total model size.  

DopantModeling was specifically developed for modeling poly(3-hexylthiophene-2,5-diyl) (P3HT), known for forming fibrillar crystallites. These fibrils are approximated as ellipsoids, with key parameters:  
- Average radius and standard deviation.  
- Minimum and maximum lengths.  

Fibrils grow until reaching the specified maximum length, encountering another fibril, or meeting a model boundary. Users can define a maximum number of fibrils. If this limit is unmet, the simulation concludes when no further fibrils can be added.  

Spatial distribution of fibrils is determined using Poisson disk sampling, which controls the average center-to-center distance. Additional options include:  
- Symmetrical growth.  
- Periodic boundary conditions.  

Like **CyRSoXS**, **DopantModeling** employs Euler angles to define orientation:  
- In-plane orientation: Random.  
- Out-of-plane orientation: Defined by a user-specified distribution. Predefined distributions, inferred from GIWAXS measurements, are provided.  

---

## Post-Processing  

After generating the fibril morphology, users can apply post-processing:  
1. Gaussian filtering to create interfacial regions.  
2. Surface roughness by specifying height features and maximum valley depth.  

Dopants can be added to the model by replacing portions of fibrillar or non-fibrillar regions. Dopant-related parameters include:  
- Total dopant amount (as a fraction of total volume).  
- Fraction of total dopant in fibrillar or non-fibrillar regions or uniformly distributed.  
- Dopant orientation in fibrillar regions (parallel, perpendicular, or random relative to fibril orientation).  

Users can specify the energy range for P-RSoXS simulations. Materials are defined by their complex X-ray refractive indices, with refractive indices available for P3HT, F4TCNQ•−, and TFSI−.  

---

## Output  

Generated morphological models can be converted to the HDF5 format required by **CyRSoXS** using **NRSS**.  

---

## Examples  

A brief example of morphology generation is provided in the `test_morphology_generation.ipynb` Jupyter notebook.  
More extensive examples are available in the publicly accessible dataset:  
[https://doi.org/10.5061/dryad.6t1g1jx65](https://doi.org/10.5061/dryad.6t1g1jx65).  

---

## Related Tools  

- **NIST RSoXS Simulation Suite (NRSS):**  
  [https://github.com/usnistgov/NRSS](https://github.com/usnistgov/NRSS)  
  NRSS includes CyRSoXS and tools for running P-RSoXS simulations.  

- **CyRSoXS:**  
  [https://github.com/usnistgov/cyrsoxs](https://github.com/usnistgov/cyrsoxs)  
  CyRSoXS is bundled with NRSS.  

---

## Citation  

If you use DopantModeling, please cite:  
*Resonant Soft X-ray Scattering Reveals the Distribution of Dopants in Semicrystalline Conjugated Polymers*  
[https://doi.org/10.1021/acs.jpcb.4c05774](https://doi.org/10.1021/acs.jpcb.4c05774)  
