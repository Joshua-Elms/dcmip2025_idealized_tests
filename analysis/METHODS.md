## Quantitative Analysis of Energy Conservation
We use Moist Non-Static Energy ($E_m$) to capture the energetic perturbation applied by ML weather models. 

Equations modified from Eqs. (1, 13, 16, 17) from [Corraliza and Mayta 2023](https://doi.org/10.1175/JAS-D-23-0005.1). 

| Term  | Description                                   | Value (if const.)   | Units                            |
|-------|-----------------------------------------------|---------------------|----------------------------------|
| $T$   | Temperature                                   |                     | $\text{K}$                       |
| $z$   | Geopotential height                           |                     | $\text{m}$                       |
| $q$   | Specific humidity                             |                     | $\text{g kg}^{-1}$               |
| $u$   | u-component of wind                           |                     | $\text{m s}^{-1}$                |
| $v$   | v-component of wind                           |                     | $\text{m s}^{-1}$                |
| $MSE$ | Specific Moist Static Energy                  |                     | $\text{J m}^{-2}$                |
| $KE$  | Specific Kinetic Energy                       |                     | $\text{J m}^{-2}$                |
| $E_m$ | Specific Moist Non-Static Energy              |                     | $\text{J m}^{-2}$                |
| $C_p$ | Specific heat of dry air at constant pressure | $1.005 \times 10^3$ | $\text{J kg}^{-1} \text{K}^{-1}$ |
| $g$   | Acceleration due to gravity                   | $9.81$              | $\text{m s}^{-2}$                |
| $L_v$ | Latent energy of vaporization                 | $2.260 \times 10^6$ | $\text{J kg}^{-1}$               |
| $R$   | Radius of Earth                               | $6.371 \times 10^6$ | $\text{m}$                       |

Moist static energy ($MSE$) is often used in atmospheric thermodynamics to track the energetic balance of large-scale systems, which should have slow enough propogation and wind speeds so as to make their kinetic energy ($KE$) terms minimal. 

$$
MSE \equiv \[C_p T + gz + L_v q\] / 4\pi R^2
$$

$$
KE \equiv \[\frac{1}{2}(u^2 + v^2)] / 4\pi R^2
$$

When considering global conservation of energy, though, $KE$ may not be neglibible, so we use the sum of the two: Moist Non-Static Energy ($E_m$). 

$$
E_m \equiv MSE + KE =  \[C_p T + gz + L_v q + \frac{1}{2}(u^2 + v^2)\] / 4\pi R^2
$$

Finally, we take the mass-weighted column integral of $E_m$ 
