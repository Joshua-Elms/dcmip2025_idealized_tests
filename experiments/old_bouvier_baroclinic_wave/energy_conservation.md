## Quantitative Analysis of Energy Conservation
We use Total energy ($E_T$) to capture the energetic perturbation applied by ML weather models. 

Equations modified from Eqs. (1, 13, 16, 17) from [Corraliza and Mayta 2023](https://doi.org/10.1175/JAS-D-23-0005.1) and Eq. (4.1) of [Darkow 1983](https://search-library.ucsd.edu/permalink/01UCS_SDI/desb83/alma991015986439706535).

|                       | Description                                   | Value (if const.)   | Units                            |
|-----------------------|-----------------------------------------------|---------------------|----------------------------------|
| $T$                   | Temperature                                   |                     | $\text{K}$                       |
| $z$                   | Geopotential height                           |                     | $\text{m}$                       |
| $q$                   | Specific humidity                             |                     | $\text{g kg}^{-1}$               |
| $u$                   | u-component of wind                           |                     | $\text{m s}^{-1}$                |
| $v$                   | v-component of wind                           |                     | $\text{m s}^{-1}$                |
| $E_T$                 | Total energy                                  |                     | $\text{J kg}^{-1}$               |
| $\langle E_T \rangle$ | Mass-weighted/column-integrated total energy  |                     | $\text{J}$                       |
| $C_p$                 | Specific heat of dry air at constant pressure | $1.005 \times 10^3$ | $\text{J kg}^{-1} \text{K}^{-1}$ |
| $g$                   | Acceleration due to gravity                   | $9.81$              | $\text{m s}^{-2}$                |
| $L_v$                 | Latent energy of vaporization                 | $2.260 \times 10^6$ | $\text{J kg}^{-1}$               |
| $R$                   | Radius of Earth                               | $6.371 \times 10^6$ | $\text{m}$                       |
| $p_s$                 | Surface pressure                              | $10^{5}$            | $\text{hPa}$                     |

Total energy combines sensible, geopotential, latent, and kinetic energies into an intensive quantity which shows energy density. 

$$
E_T \equiv C_p T + gz + L_v q + \frac{1}{2}(u^2 + v^2)
$$

We take the mass-weighted column integral of $E_T$ to calculate $\langle E_T \rangle$, the extensive total energy of a column. 

$$
\langle E_T \rangle = \frac{1}{g}\int_0^{p_s} E_T dp
$$

References: 

- Adames Corraliza, Ángel F., and Víctor C. Mayta. "On the Accuracy of the Moist Static Energy Budget When Applied to Large-Scale Tropical Motions". Journal of the Atmospheric Sciences 80.10 (2023): 2365-2376. https://doi.org/10.1175/JAS-D-23-0005.1 Web.
- Darkow, G. L., 1983:  Basic thunderstorm energetics and thermodynamics.  "Thunderstorm Morphology and Dynamics," E. Kessler, Ed., 2nd ed., University of Oklahoma Press, 59-73. https://search-library.ucsd.edu/permalink/01UCS_SDI/desb83/alma991015986439706535
