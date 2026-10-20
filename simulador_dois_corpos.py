import numpy as np
import pandas as pd
import rebound
import matplotlib.pyplot as plt

##################################################################
# PARAMETROS DE ENTRADA
# constantes, SI
G    = 6.67430e-11      # constante gravitacional
YEAR = 365.25 * 86400.0 # 1 ano em segundos
AU   = 1.495978707e11   # unidade astronomica, m

# corpo primario (central)
M = 1.989e30            # massa do primario, kg

# corpo secundario - elementos orbitais osculadores em t = 0
m     = 5.972e24        # massa do secundario, kg
sma   = 1.0 * AU        # semieixo maior, m
ecc   = 0.0167          # excentricidade
inc   = np.radians(0.0) # inclinacao, rad
Omega = np.radians(0.0) # longitude do nodo ascendente, rad
omega = np.radians(102.9) # argumento do pericentro, rad
f     = np.radians(0.0) # anomalia verdadeira inicial, rad

# controle da integracao
N_ORBITS    = 3         # numero de orbitas a integrar
N_OUT       = 2000      # numero de pontos de saida
MOVE_TO_COM = True

# arquivos de saida
FILE_STATE = 'estado.dat'
FILE_ORBIT = 'elementos_orbitais.dat'

##################################################################
# INTEGRACAO VIA REBOUND

sim = rebound.Simulation()
sim.G = G

sim.add(m=M)
sim.add(m=m, a=sma, e=ecc, inc=inc, Omega=Omega, omega=omega, f=f)

if MOVE_TO_COM:
    sim.move_to_com()

# periodo orbital via 3a lei de kepler
mu = sim.G * (M + m)
period = 2 * np.pi * np.sqrt(sma**3 / mu)
t_max = N_ORBITS * period
times = np.linspace(0.0, t_max, N_OUT)

print(f'periodo orbital do secundario : {period/YEAR:.4f} anos')
print(f'tempo total de integracao     : {t_max/YEAR:.4f} anos')
print(f'numero de pontos de saida     : {N_OUT}')
print('integrando...')

state_rows = []
orbit_rows = []

primary = sim.particles[0]
secondary = sim.particles[1]

step_print = max(1, N_OUT // 10)

for i, t in enumerate(times):
    sim.integrate(t)

    # coordenadas relativas (m)
    x = secondary.x - primary.x
    y = secondary.y - primary.y
    z = secondary.z - primary.z

    orb = secondary.orbit(primary=primary)
    t_years = sim.t / YEAR

    state_rows.append([t_years, x, y, z])
    orbit_rows.append([
        t_years, orb.a, orb.e, 
        np.degrees(orb.inc),
        np.degrees(orb.Omega), 
        np.degrees(orb.omega),
        np.degrees(orb.f)
    ])

    if (i + 1) % step_print == 0 or i == N_OUT - 1:
        pct = ((i + 1) / N_OUT) * 100
        print(f'tempo: {t_years:.3f} anos ({pct:.0f} %)')

cols_state = ['time', 'x', 'y', 'z']
cols_orbit = ['time', 'a', 'e', 'inc', 'Omega', 'omega', 'f']

df_state = pd.DataFrame(state_rows, columns=cols_state)
df_orbit = pd.DataFrame(orbit_rows, columns=cols_orbit)

# salva arquivos
df_state.to_csv(FILE_STATE, sep='\t', index=False, float_format='%1.10e')
df_orbit.to_csv(FILE_ORBIT, sep='\t', index=False, float_format='%1.10e')

##################################################################
# PLOTAGEM GRAFICA (MATPLOTLIB)

# Converter posicoes para Unidades Astronomicas (AU) nos graficos
x_au = df_state['x'] / AU
y_au = df_state['y'] / AU
z_au = df_state['z'] / AU
t_yr = df_state['time']

# -------------------------------------------------------------
# FIGURA 1: Trajetoria 3D e Projecao 2D (Plano Orbital XY)
# -------------------------------------------------------------
fig1 = plt.figure(figsize=(13, 5))

# Subplot 1: Trajetoria 3D
ax3d = fig1.add_subplot(1, 2, 1, projection='3d')
ax3d.plot(x_au, y_au, z_au, label='Trajetória Secundário', color='tab:blue', lw=1.2)
ax3d.scatter([0], [0], [0], color='gold', s=120, label='Primário (Origem)', edgecolors='black')
ax3d.scatter([x_au.iloc[0]], [y_au.iloc[0]], [z_au.iloc[0]], color='red', s=40, label='Início ($t=0$)')

ax3d.set_xlabel('X [AU]')
ax3d.set_ylabel('Y [AU]')
ax3d.set_zlabel('Z [AU]')
ax3d.set_title('Órbita Relativa em 3D')
ax3d.legend(loc='upper right', fontsize=8)

# Subplot 2: Projecao 2D no plano XY
ax2d = fig1.add_subplot(1, 2, 2)
ax2d.plot(x_au, y_au, color='tab:blue', label='Órbita (Plano XY)', lw=1.2)
ax2d.scatter([0], [0], color='gold', s=120, label='Primário', edgecolors='black', zorder=5)
ax2d.scatter([x_au.iloc[0]], [y_au.iloc[0]], color='red', s=40, label='Início ($t=0$)', zorder=5)

ax2d.set_xlabel('X [AU]')
ax2d.set_ylabel('Y [AU]')
ax2d.set_title('Projeção no Plano Orbital ($XY$)')
ax2d.set_aspect('equal', 'datalim')
ax2d.grid(True, linestyle='--', alpha=0.5)
ax2d.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('orbita_2d_3d.png', dpi=300)

# -------------------------------------------------------------
# FIGURA 2: Evolucao dos Elementos Orbitais Osculadores
# -------------------------------------------------------------
fig2, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
fig2.suptitle('Evolução dos Elementos Orbitais Osculadores', fontsize=14)

# (a) Semieixo maior
axs[0, 0].plot(t_yr, df_orbit['a'] / AU, color='tab:red')
axs[0, 0].set_ylabel('$a$ [AU]')
axs[0, 0].grid(True, linestyle='--', alpha=0.5)

# (e) Excentricidade
axs[0, 1].plot(t_yr, df_orbit['e'], color='tab:orange')
axs[0, 1].set_ylabel('$e$')
axs[0, 1].grid(True, linestyle='--', alpha=0.5)

# (inc) Inclinacao
axs[1, 0].plot(t_yr, df_orbit['inc'], color='tab:green')
axs[1, 0].set_ylabel('$i$ [graus]')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)

# (Omega) Longitude do Nodo Ascendente
axs[1, 1].plot(t_yr, df_orbit['Omega'], color='tab:purple')
axs[1, 1].set_ylabel('$\Omega$ [graus]')
axs[1, 1].grid(True, linestyle='--', alpha=0.5)

# (omega) Argumento do Pericentro
axs[2, 0].plot(t_yr, df_orbit['omega'], color='tab:brown')
axs[2, 0].set_ylabel('$\omega$ [graus]')
axs[2, 0].set_xlabel('Tempo [anos]')
axs[2, 0].grid(True, linestyle='--', alpha=0.5)

# (f) Anomalia Verdadeira
axs[2, 1].plot(t_yr, df_orbit['f'], color='tab:blue', lw=0.8)
axs[2, 1].set_ylabel('$f$ [graus]')
axs[2, 1].set_xlabel('Tempo [anos]')
axs[2, 1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('elementos_orbitais.png', dpi=300)
plt.show()

print('Imagens "orbita_2d_3d.png" e "elementos_orbitais.png" geradas com sucesso!')