# CHIP
Cosmic Horizons In Python<br>
The Hubble parameter has the form
$$H \equiv \frac{\dot{a}}{a}$$ where $a(t)$ is the scale factor defined via the Friedmann equations. The model in operation here is the $\Lambda$ CDM ($\Lambda$ Cold Dark Matter) model. The current epoch is at $a=1$ and $a(t)$ is an increasing function of $t$. The universal acceleration $\ddot{a} \gt 0$ but $H(a)$ is <i>decreasing</i>.<br><br>The Hubble parameter is given by<br><br>$$H(a)=H_0\sqrt{\Omega_r a^{-4}+\Omega_m a^{-3}\Omega_k a^{-2}+\Omega_\Lambda}$$<br><br>Where $\Omega_r$ is the radiation density, $\Omega_m$ is the matter density and $\Omega_\Lambda$ is the dark energy density - all values at the current epoch. $\Omega_k$ is a parameter relating to the curvature of space and can be found as $\Omega_k=1-\Omega_r-\Omega_m-\Omega_\Lambda$. $H_0$ is the value of the Hubble parameter at the present epoch - the Hubble constant.<br><br>The <b>cosmic time</b> can be found from $$\dot{a}=\frac{da}{dt}\implies dt = \frac{da}{\dot{a}}=\frac{da}{aH(a)}\implies t(a)=\int_0^a \frac{1}{aH(a)} da$$<br><br>If $\rho$ is a length comoving with the expansion the for null geodesics we have $$0=dt^2-a(t)^2d\rho^2$$ so the <i>comoving</i> distance travelled by a photon is given by<br><br>$$r_c=c\int_{a_1}^{a_2}\frac{1}{a\dot{a}}da=c\int_{a_1}^{a_2}\frac{1}{a^2H(a)}da$$<br><br>where $a_1$ is the scale factor at epoch of emisson and $a_2$ is scale factor at epoch of reception of the photon.<br><br>The <i>proper</i> distance is related to the comoving distance by <br><br>$$r = a(t)r_c$$<br><br>![Screenshot 2024-10-30 091202](https://github.com/user-attachments/assets/153a4236-f1f9-4395-8748-ffcc23b95138)
<br>The <b>light cone</b> is all events that we can currently observe and so are on our past light cone (the cone or 'teardrop' with apex at a(t) = 1). Note that eventually the photon begins to lose ground to the expansion as the expansion is greater than the velocity of the photon. This is essentially a plot of cosmic time against $r$.<br><br>
The <b>particle horizon</b> is the maximum distance that a photon has been able to travel to us between $a=0$ and the present epoch $a=1$. It is as far as we can theoretically see - the edge of the observable universe and is given by<br><br>$$r_p=c\int_0^1 \frac{1}{a^2H(a)}da$$<br><br>The particle horizon intersect the $a=1$ line at about 46 Gly.<br><br>The <b>event horizon</b>  is the region of space from which a photon that is emitted at $a=1$ will still be able to reach us at some point in the future and is given by<br><br>
$$r_e=c\int_1^\infty \frac{1}{a^2H(a)}da$$<br><br>
The event horizon intersects the $a=1$ line at about 17 Gly.<br><br>The <b>Hubble radius</b> although not strictly a horizon is given by<br><br>$$r_h=\frac{c}{H(a)}$$<br><br>It defines the boundary between particles that are moving with sub and super luminal speeds relative to an observer at one given time. The Hubble radius at the present epoch is about 14.5 Gly.<br><br>The spacetime plot in co-moving coordinates is given below
![Screenshot 2024-10-30 115255](https://github.com/user-attachments/assets/078f48ba-e27a-40e4-b5f4-3b6c32913872)
