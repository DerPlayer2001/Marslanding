# Marslanding

This project is a simulation of a mars landing.

Demo Hostet at [mars.hobbylos.org](https://mars.hobbylos.org).

To run the project locally just start the docker compose file with `docker compose up` and open the browser at [localhost:80](http://localhost:80).

## Math

### Force over time with changing mass

$\displaystyle \Delta v =  F * \frac{t}{m}$ (Change in velocity)

$\displaystyle \Delta t$: Time Interval  
$\displaystyle m_{0}$: Mass at $\displaystyle t_{0}$  
$\displaystyle \Delta m$: Change in Mass in Time Interval $\displaystyle \Delta t$

$\displaystyle m(t) =  m_{0} - \frac{\Delta m * t}{\Delta t}$ (Linear Equation)

$\displaystyle  \Delta v = 
\int_{0}^{\Delta t} \frac{F}{m_{0} - \frac{\Delta m * t}{\Delta t}} \,dt = \\
-F {\left(\frac{\Delta t \log(-\Delta m + m_{0})}{\Delta m} - \frac{\Delta t \log(m_{0})}{\Delta m}\right)} = \\
F {\left(\frac{\Delta t \log(m_{0})}{\Delta m} - \frac{\Delta t \log(m_{0} - \Delta m)}{\Delta m}\right)} = \\
F * \frac{\Delta t}{\Delta m} {(\log(m_{0}) - \log(m_{0} - \Delta m))} = \\
F * \frac{\Delta t}{\Delta m} * \log\left(\frac{m_{0}}{m_{0}-\Delta m}\right)$

[Calculation](./Math.ipynb)

## Sources

[(1) Barometrische Höhenformel [Zitat vom 10.06.2024]](https://de.wikipedia.org/wiki/Barometrische_Höhenformel)  
[(2) Schwerefeld [Zitat vom 10.06.2024]](https://de.wikipedia.org/wiki/Schwerefeld)  
[(3) Gravitationskonstante [Zitat vom 10.06.2024]](https://de.wikipedia.org/wiki/Gravitationskonstante)  
[(4) Drag [Zitat vom 10.06.2024]](<https://en.wikipedia.org/wiki/Drag_(physics)>)  
[(5) Specific Impulse [Zitat vom 10.06.2024]](https://en.wikipedia.org/wiki/Specific_impulse)  
[(6) Barometrische Höhenformel für die Marsatmosphäre [Zitat vom 11.06.2024]](https://rmtux.de/?q=node/147)

## Image-Sources

[(1) Bild des Raumschiffs [Abgerufen am 11.06.2024]](https://de.cleanpng.com/png-t28orl/)  
[(2) Bild der Flamme [Abgerufen am 11.06.2024]](https://de.cleanpng.com/png-z747a0/)
