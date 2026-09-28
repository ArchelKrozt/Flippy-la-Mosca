# 🪰 FLIPPY LA MOSCA

**El cerebro completo de una mosca de la fruta, corriendo en tu portátil y viviendo en un mundo pixel-art.**

Flippy tiene el conectoma real de *Drosophila melanogaster* (FlyWire v783: **138.639 neuronas y 15 millones
de conexiones**) simulado neurona a neurona. Lo que ve, huele, oye, toca y saborea en la arena entra en sus
neuronas sensoriales; lo que hacen sus neuronas motoras mueve su cuerpo. Nadie le programó que coma, que huya
o que elija pareja: lo decide su conectoma.

![Flippy en la arena](docs/demo.gif)

Funciona en un **MacBook Air M1 con 8 GB** a ≈ tiempo real (~1,5 GB de RAM).

## Probarlo

**Necesitas:** Python 3.10 o superior, ~2 GB de RAM libre, ~500 MB de disco e internet la primera vez (para
descargar el conectoma). Probado en macOS (Apple Silicon); en Linux y Windows debería funcionar, pero no se ha probado.

### La forma fácil (macOS / Linux)
Haz doble clic en **`iniciar_flippy.command`** (en Linux: `./iniciar_flippy.command`). La primera vez prepara un
entorno de Python dentro de la carpeta, instala lo necesario y descarga el conectoma (unos minutos); después abre el
juego en el navegador. Las siguientes veces abre directamente.
Si macOS dice que no puede abrirlo por ser de un desarrollador no identificado: clic derecho → **Abrir** → **Abrir**.

### Con la Terminal
macOS / Linux:
```bash
python3 -m venv .venv              # entorno propio (evita el error "externally-managed-environment")
source .venv/bin/activate
pip install -r requirements.txt    # numpy, pandas, pyarrow, numba
python download_data.py            # ~135 MB del conectoma, una sola vez
python server.py                   # abre http://localhost:8000
```
Windows (PowerShell):
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python server.py
```
Las siguientes veces basta con activar el entorno (`source .venv/bin/activate`) y `python server.py`.

### Opciones
- Al arrancar, `server.py` muestra un **menú de conectomas**: los descargados y otros que puedes descargar
  (FlyWire v630, Hemibrain, MANC, BANC, Male CNS). Hoy solo FlyWire v783 funciona en el juego; los demás se
  guardan en `data/models/` para hacerlos compatibles más adelante. Enter = jugar.
- `python server.py --no-menu` entra directo · `--port 8080` usa otro puerto · `--no-browser` no abre el navegador.
- `python download_data.py --list` lista el catálogo; `python download_data.py manc121` descarga uno sin abrir el juego.
- `python -m flippy.selftest` comprueba en ~20 s que el cerebro reproduce los resultados clave (9 pruebas).

### Apagar y guardar
- Botón **⏻ GUARDAR Y APAGAR** en la página, o **Ctrl+C** en la Terminal: en ambos casos guarda antes de cerrar.
- La partida se **autoguarda** cada minuto; al volver a abrir te ofrece **▶ CONTINUAR**. También puedes guardar y
  cargar partidas con nombre (💾 / 📂). Las partidas están en la carpeta `saves/`.

## Guía de la interfaz

- **Arena:** elige una herramienta y haz clic en la arena para colocarla: 🍌 comida, ☠ amargo, 💨 CO₂, 🔥 calor,
  🌀 viento, 💡 luz, 🪨 piedra, 🍂 hoja, ✋ manotazo, 🪰 muñeca (abre una ventana para configurarla) y ✖ borrar.
- **Controles:** pausa, reiniciar, guardar/cargar, ⏻ guardar y apagar, 🌙 ciclo día/noche, tamaño de la arena,
  **velocidad** de la simulación y **ganas de caminar** del cuerpo (− más quieta, + camina más).
- **Ficha de Flippy** (derecha): su estado, energía, **época de apareamiento** (botón: actívala para que acepte
  machos) y **navegación asistida** (botón: desactívala para ver el comportamiento 100 % cerebral).
- **PANELES** (ventanas flotantes que se arrastran y recuerdan su posición; se cierran con ✖ o Esc):
  🪰 Flippy en detalle · 🧪 Experimento: silenciar neuronas · 🧭 Memoria de Flippy · 👁 Lo que ve Flippy ·
  🧠 Cerebro (disparos en vivo) · 📶 Sentidos → neuronas · ⚡ Órdenes al cuerpo. Cada botón muestra un resumen en vivo.
- **Moscas muñeca** (abajo): la lista, **🕸 RELACIONES** (grafo social y recuerdos de Flippy) y **EDITAR** cada una.
- **Árbol familiar** (abajo): con el botón **📜 HISTORIA** ves la cronología completa y puedes seguir a una mosca concreta.

![Interfaz](docs/ui.png)

## Qué puedes hacer

| Herramienta | Qué pasa en el cerebro de Flippy |
|---|---|
| 🍌 Comida | huele a fruta (ORN DM1/DM4…) → se acerca; al tocarla, las neuronas del azúcar activan **MN9** y saca la probóscide para comer |
| ✋ Manotazo | una sombra que crece activa **LC4/LPLC2** → **fibra gigante (DNp01)** → despega para escapar |
| 💨 CO₂ · 🔥 Calor | activan **DNb05** → se aleja; el calor además es un castigo que recuerda |
| ☠ Amargo | activa sus neuronas del amargo (patas); en este modelo apenas llega a las órdenes motoras |
| 🌀 Viento | activa el órgano de Johnston (antenas); se ve en sus sentidos, pero en este modelo no cambia su conducta |
| 🪨 Piedra | un obstáculo que se acerca activa **LPLC1** de ese ojo → el cerebro gira hacia el lado contrario (con LPLC1 pasa un 64 % menos de tiempo chocando) |
| 🍂 Hoja | refugio en sombra: las moscas duermen debajo y la vista llega atenuada |
| 💡 Luz | los **ocelos** activan **DNp18** (más luz, más disparo) → va hacia el ojo más iluminado (asistido) |
| 🪰 Mosca muñeca | compañeras configurables (sexo, época de apareamiento, violencia, ayudante, sociable, canto, insistencia…) |

Las muñecas tienen comportamiento programado; solo Flippy tiene cerebro. Le llegan únicamente por sus sentidos:

| La muñeca… | Sentido de Flippy | Neuronas | Respuesta del conectoma |
|---|---|---|---|
| está a la vista | ojo izq./der. | LC10a / LC11 | gira hacia ella (DNa02 del mismo lado) |
| embiste rápido | sombra que crece | LC4 / LPLC2 | escape por la fibra gigante (DNp01) |
| la toca | antenas | mecanorreceptores | se acicala (DNg84) |
| está cerca | su olor individual | 10 glomérulos → células de Kenyon | la recuerda (ver *Memoria real*) |
| es macho | feromona cVA | ORN DA1 (Or67d) | |
| canta (macho en celo) | canto | JO-B + vpoEN\* | si está en época (pC1) → vpoDN → acepta |

### Sustos graduales, sueño y refugio
- **Escape gradual con habituación:** una descarga fuerte de la fibra gigante (DNp01) la hace despegar; una
  alarma más leve (DNp02/04/11) solo la hace alejarse caminando. Cada despegue sube el umbral (habituación) y en
  ~20 s de calma se recupera. Las moscas descuentan su propio movimiento: solo "asusta" lo que se les acerca.
- **🌙 Día y noche** (2,5 min + 1,5 min): de noche las moscas duermen, mejor bajo una **🍂 hoja**. Dormida, sus
  sentidos llegan atenuados (umbral de despertar alto); la despiertan la fibra gigante, la alarma, el tacto o el
  amanecer. Bajo una hoja la vista llega en sombra. Durante el sueño la memoria no se olvida (consolidación).
  El sueño es un estado interno: el conectoma no lo genera, pero lo que la despierta sí sale de él.

### Espacio
Tres tamaños de arena (pequeña 320×200, mediana 480×300, grande 640×400), que se cambian en cualquier momento.
La población máxima depende del área (~1 mosca por 6.400 px²: 10 / 22 / 40). El tacto es fásico, como los
mecanorreceptores reales: un choque se siente entero y el contacto sostenido (caminar pegada a una pared) se adapta
en ~0,6 s; las muñecas respetan un espacio personal alrededor de Flippy.

### 👁 Lo que ve Flippy
Una ventana muestra su panorama de ~300° en primera persona (suelo en perspectiva, paredes, piedras, moscas,
comida, luz y la mano que se acerca) y qué neuronas detectoras se activan en cada ojo. Al cerebro no le llega
esa imagen píxel a píxel, sino a través de sus detectores: LC4/LPLC2 (algo que se acerca), LC10a/LC11 (otra
mosca), LPLC1 (obstáculo) y los ocelos (luz). Probado y descartado por ahora: las células HS/VS (sentir su propio
giro) y los fotorreceptores R1-6 no llegan a las órdenes de caminar en este modelo.

**Hallazgo: la visión desde los fotorreceptores no funciona en este modelo.** Proyectamos una sombra que crece sobre
los ~4.000 fotorreceptores R1-6 de un ojo, colocados en su posición real (su disposición retinotópica se reconstruye
de la anatomía: casi un plano). Resultado: la señal muere en la médula. Con luz intensa se activa el 20 % de la
lámina (L1-L3) y el 4 % de las Tm, pero nada llega a T4/T5 (movimiento), a LC4/LPLC2 (amenazas) ni a la fibra
gigante; ni con el modelo tal cual ni corrigiendo el signo de los fotorreceptores (en FlyWire se predicen
colinérgicos, en realidad liberan histamina, inhibidora). La causa es de fondo: las primeras capas de la retina
funcionan con voltajes graduales, detectan la oscuridad por desinhibición y calculan el movimiento con retardos
finos, algo que un modelo de neuronas que disparan con un único tipo de sinapsis no reproduce. Los trabajos que sí
lo logran (Lappalainen et al. 2024, *Nature*) entrenan los parámetros de cada tipo celular. Por eso la visión de
Flippy entra por sus detectores (LC4, LPLC2, LC10a, LC11, LPLC1, ocelos): desde ahí todo es conectoma.

### Cortejo: Flippy elige
Los machos siguen el ritual real: orientarse → tocarla → **cantar con un ala** → intentar la cópula. La decisión
es del conectoma: **vpoDN** (acepta) frente a **DNp13** (rechaza sacando el ovipositor). Medido en el modelo: más
canto → más vpoDN (4 → 13 Hz), así que el buen cantor convence antes; sin el impulso pC1 (fecundada o fuera de
época) el canto activa DNp13, como en Wang et al. 2021. Si dos machos cortejan a la vez, pelean. Un macho que se
acerca demasiado rápido dispara la fibra gigante de Flippy: el buen cortejo es lento.

### Descendencia y árbol familiar
Tras la cópula maduran huevos y Flippy pone cuando lo decide su **oviDN**. El azúcar y el olor a comida lo frenan
(−20 % / −35 %), la misma dirección que Yang et al. 2008 (las hembras evitan poner sobre sacarosa). Huevo → larva
(busca comida) → pupa → adulto → vejez, en ~3 minutos; las crías heredan una mezcla de los rasgos y del olor de sus
padres. Tras copular, cada mosca descansa una generación; las muñecas mueren tras su tercera cópula. Las muñecas
también se reproducen entre ellas, así que aparecen generaciones, con **árbol genealógico y cronología**.

![Historia familiar](docs/history.png)

### Memoria real
Cada mosca tiene un **olor individual** (10 glomérulos) que activa ~1 % de las 5.177 células de Kenyon, con poco
solapamiento entre moscas. Las 62.261 sinapsis Kenyon→MBON son **plásticas** (regla de Hige et al. 2015): Kenyon
activa + dopamina en su compartimento → la sinapsis se debilita; oler sin dopamina → extinción lenta. Qué dopamina
inerva cada MBON se lee del conectoma y coincide con el mapa de Aso et al. 2014.
- Un golpe → dolor → dopamina **PPL1** → debilita las MBON de "acercarse" para ese olor → la evita.
- Comer con alguien cerca → dopamina **PAM** → debilita las MBON de "evitar" → se acerca.

Condicionamiento en el modelo: olor + dolor → −1,00; olor solo → +0,07; olor + recompensa → +1,00. La memoria se
guarda con la partida.

**Memoria de lugares** con el mismo mecanismo: cada hoja, comida y zona de calor tiene su propio olor. Comer
en un sitio → le gusta; pasar calor → lo evita (el calor es un castigo, PPL1); dormir tranquila bajo una hoja →
"alivio" (PAM) → la prefiere. Medido: tras comer en un comedero, con hambre vuelve a él desde 358 px, fuera del
alcance de su olor (sin esa experiencia, no); tras dormir bien bajo una hoja, a la noche siguiente duerme en ella
aunque haya otra más cerca. Ventana **🧭 MEMORIA DE FLIPPY**: minimapa, listas y lo que va aprendiendo.

![Relaciones](docs/relations.png)

## 🧪 Modo experimento: silenciar neuronas

Como en un laboratorio: las neuronas silenciadas **no pueden disparar** (el equivalente a la optogenética con
GtACR1 o a expresar el canal Kir2.1), así que todo lo que dependía de ellas deja de recibir su señal. Ventana
**🧪 EXPERIMENTO** en PANELES:

1. **Por función:** un catálogo documentado; cada botón explica qué es, dónde está, qué pasará y la referencia.
2. **Haciendo clic en el mapa del cerebro** (vista frontal): apaga una zona, como un láser. Al pasar el ratón, un
   tooltip dice cuántas neuronas hay debajo, de qué regiones, los tipos principales explicados y qué funciones del
   juego se verían afectadas. Clic en una zona ya silenciada = restaurarla.
3. **Silenciadas ahora:** lista para restaurar una a una o todas. Las zonas se ven en rojo en la ventana del cerebro.
   Los experimentos se guardan con la partida y quedan en la cronología.

| Función | Dónde | Si la silencias | Referencia |
|---|---|---|---|
| Fibra gigante | cerebro → médula ventral (2 neuronas) | Ante una amenaza ya no despega: como mucho se aleja caminando (neuronas de alarma). | von Reyn et al. 2014 |
| Comer (MN9) | ganglio subesofágico (7 neuronas) | Aunque pise la comida y sus neuronas del azúcar disparen, no come y su energía baja. | Shiu et al. 2024 |
| Girar (DNa01/DNa02) | descendentes (4 neuronas) | Deja de girar por decisión de su cerebro: no sigue a otras moscas ni esquiva obstáculos. | Rayshubskiy et al. 2020 |
| Detectar amenazas (LC4/LPLC2) | lóbulo óptico → cerebro (~310) | Queda 'ciega' a las amenazas: un manotazo ya no la asusta. | Ache et al. 2019 |
| Esquivar obstáculos (LPLC1) | lóbulo óptico → cerebro (~140) | Choca más con piedras y paredes. | Tanaka & Clark 2022 |
| Ver otras moscas (LC10a/LC11) | lóbulo óptico → cerebro (~360) | Ya no se gira hacia las moscas que ve. | Ribeiro et al. 2018 |
| Memoria (células de Kenyon) | cuerpo pedunculado (5.177) | No aprende nada nuevo y no puede usar lo que recordaba (no reconoce olores). | Heisenberg 2003 |
| Aprender castigos (dopamina PPL1) | cuerpo pedunculado (16) | Deja de aprender de los golpes y del calor; sigue aprendiendo lo bueno. | Aso et al. 2014 |
| Aprender recompensas (dopamina PAM) | cuerpo pedunculado (~300) | Deja de aprender lo bueno (comederos, hojas favoritas); sigue aprendiendo castigos. | Aso et al. 2014 |
| Aceptar pareja (vpoDN) | descendentes (2 neuronas) | Nunca acepta a un macho, aunque esté en época y él cante muy bien. | Wang et al. 2021 |
| Rechazar (DNp13) | descendentes (2 neuronas) | Ya no puede rechazar con el ovipositor. | Wang et al. 2021 |
| Deseo sexual (pC1) | protocerebro (10 neuronas) | Aunque la pongas en época, no llega a estar receptiva. | Zhou et al. 2014 |
| Poner huevos (oviDN) | descendentes (6 neuronas) | Madura huevos pero no los pone. | Wang et al. 2020 |
| Acicalarse (DNg84) | descendentes (2 neuronas) | Ya no se acicala al tocarla. | Hampel et al. 2015 |
| Atracción por el olor (DNg100/DNge053) | descendentes (4 neuronas) | El olor a comida deja de atraerla (solo la encuentra por azar). | este proyecto |
| Oler la fruta (ORN) | antenas (~170 neuronas) | Anosmia a la comida: no la huele. | Semmelhack & Wang 2009 |
| Notar la luz (ocelos) | ocelos → cerebro (20) | La lámpara deja de atraerla. | Hengstenberg 1993 |
| Freno del olfato (interneuronas GABA) | lóbulo antenal (~150) | Medido en este modelo: las neuronas de proyección del olfato responden un 20-30 % más fuerte al mismo olor (menos contraste entre olores). No llega a provocar una convulsión. | Olsen & Wilson 2008 |

Verificado en el modelo: sin MN9 no come aunque pise la comida (energía −3 frente a +38 en 5 s); sin la fibra gigante
un manotazo no la hace despegar; sin células de Kenyon deja de reconocer a quien recordaba; sin el freno GABA del
lóbulo antenal sus neuronas de proyección olfativa responden un 20-30 % más al mismo olor.

## Qué es cerebro y qué no

**Del conectoma:** comer (MN9), escapar (DNp01), acicalarse (DNg84), girar hacia otra mosca, aceptar o rechazar
(vpoDN / DNp13), cuándo poner huevos (oviDN), qué aprende la memoria (Kenyon→MBON).

**Añadido o asistido** (marcado en la interfaz):
- **Navegación asistida** (interruptor): en este modelo el *lado* de un olor no llega a las neuronas descendentes
  (medido: DNa01/02 disparan igual con olor a la izquierda o a la derecha). El cerebro decide *si* le atrae un olor,
  una mosca o un recuerdo; un reflejo del cuerpo elige *hacia dónde*. Con olfato asistido llega a la comida 8/8
  veces en ~2,4 s; sin olfato, 1/8 en 17 s. Desactívala para ver el comportamiento 100 % cerebral.
- **Estados internos como estímulos:** época de apareamiento (pC1), huevos maduros (SMP550), dolor (PPL1) y
  recompensa (PAM). En la mosca real llegan por hormonas o por la médula ventral, que no están en este conectoma.
- \***Atajo del canto:** el canto también estimula vpoEN; el filtro temporal del pulso de canto no se reproduce con
  pesos fijos (el canto solo nunca llega a vpoEN en simulación).
- **Ganas de caminar:** el conectoma no genera marcha espontánea; el cuerpo camina y el cerebro lo modula.
- La "opinión" de la memoria se calcula de las sinapsis actuales (la entrada que recibirían las MBON); el cambio
  en el disparo de las MBON es pequeño porque cada olor activa pocas Kenyon.

**Cambios al modelo original de Shiu et al.:**
- Las interneuronas locales del lóbulo antenal con neurotransmisor *predicho* dopamina/serotonina son en realidad
  GABAérgicas: sin esta corrección, cualquier olor o calor provocaba una "convulsión" de ~10.000 neuronas saturadas.
- Adaptación por frecuencia de disparo (`a_inc=0.5`, `tau_a=300 ms`), para que el circuito no se quede enganchado
  en estados autosostenidos.
- Las neuronas cuyo potencial está a menos de 0,2 mV del reposo no se integran (resultado idéntico en las pruebas,
  mucho más rápido).

## Cómo funciona

| Archivo | |
|---|---|
| `flippy/brain.py` | Modelo LIF de Shiu et al. 2024, *event-driven* (solo integra neuronas fuera de reposo) y compilado con numba; sin numba usa numpy |
| `flippy/neurons.py` | Qué tipos celulares de FlyWire son sentidos y cuáles se leen como órdenes motoras. Edítalo para recablear el cuerpo |
| `flippy/memory.py` | Plasticidad dopaminérgica del cuerpo pedunculado |
| `flippy/world.py` | Arena, cuerpo, reproducción, genealogía, guardado |
| `flippy/dolls.py` | Moscas muñeca: rasgos, cortejo, relaciones |
| `flippy/selftest.py` | Pruebas de los resultados clave |
| `server.py`, `web/index.html` | Servidor (solo biblioteca estándar de Python) e interfaz |
| `flippy/experiments.py` | Catálogo documentado del modo experimento (silenciar neuronas) |
| `flippy/catalog.py` | Catálogo de conectomas descargables (menú de inicio) |
| `download_data.py` | Descarga del conectoma |
| `iniciar_flippy.command` | Lanzador de doble clic (macOS/Linux) |

Cada tick (20 ms de tiempo cerebral): mundo → tasas sensoriales (Poisson) → 100 pasos de 0,2 ms de las 138.639
neuronas → descendentes → cuerpo.

## Problemas frecuentes

- **"El puerto 8000 está ocupado"**: ya hay un FLIPPY abierto (ciérralo con su botón ⏻), o usa `--port 8001`.
- **"externally-managed-environment" al instalar**: usa un entorno propio (`python3 -m venv .venv`, ver arriba).
- **macOS no deja abrir `iniciar_flippy.command`**: clic derecho → Abrir → Abrir.
- **Va lento**: la simulación usa un núcleo de la CPU; con muchas muñecas pegadas a Flippy baja a ~0,7× tiempo
  real. Cerrar la ventana 🧠 Cerebro y reducir muñecas ayuda.
- **Empezar de cero**: botón REINICIAR (el autoguardado se sobrescribe en un minuto), o borra la carpeta `saves/`.
- **Rehacer la instalación**: borra la carpeta `.venv/` (y `data/` si quieres volver a descargar los datos).

## Créditos y citas

- **Modelo LIF:** Shiu, P. K. et al. (2024). *A Drosophila computational brain model reveals sensorimotor
  processing.* Nature. Código y conectividad: [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
- **Conectoma FlyWire:** Dorkenwald, S. et al. (2024). *Neuronal wiring diagram of an adult brain.* Nature.
- **Tipos celulares:** Schlegel, P. et al. (2024). *Whole-brain annotation and multi-connectome cell typing of
  Drosophila.* Nature. [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
  (para versiones ≥ 3.0 piden citar también Matsliah et al. 2024 y Berg et al. 2025).
- Plasticidad: Hige, T. et al. (2015), *Neuron*. Compartimentos del cuerpo pedunculado: Aso, Y. et al. (2014), *eLife*.
- Receptividad sexual: Wang, F. et al. (2021), *Neuron*. Elección del sitio de puesta: Yang, C. et al. (2008), *Science*.

El código de este proyecto es MIT (ver `LICENSE`). Los datos del conectoma pertenecen a sus autores: consulta las
condiciones de uso de FlyWire antes de redistribuirlos.
