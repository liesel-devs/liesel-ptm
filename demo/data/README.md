# Columbus example data

`columb.csv` and `columb_polys.csv` are CSV exports of the `columb` and
`columb.polys` datasets shipped with R's **mgcv** package, copied from
`liesel_gam/tests/mgcv_data`. They let `term-distribution-plots.ipynb` run without
an R installation or a `liesel_gam` source checkout.

`columb.csv` contains 49 districts. In `columb_polys.csv`, `label` identifies the
district and `vertex` orders its polygon coordinates; missing coordinates
separate polygon boundaries.

The mgcv documentation credits the original data to Anselin, Luc (1988),
*Spatial econometrics: methods and models*, Kluwer Academic, Table 12.1, p. 189,
via the `columbus` example in **spdep**.
