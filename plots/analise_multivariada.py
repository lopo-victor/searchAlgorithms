import pandas as pd
import matplotlib.pyplot as plt
from math import pi

df = pd.read_csv("resultados.csv")

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for player in df['player'].unique():
    dados = df[df['player'] == player]
    ax.scatter(dados['score'], dados['steps'], dados['battery'], label=player)
ax.set_xlabel('Pontuação')
ax.set_ylabel('Passos')
ax.set_zlabel('Bateria')
ax.set_title('Score vs Steps vs Battery')
ax.legend()
plt.tight_layout()
plt.savefig("3d_score_steps_battery.png")
plt.show()




metrics = ['score', 'steps', 'deliveries', 'battery']
radar_df = df.groupby('player')[metrics].mean().reset_index()

radar_normalized = radar_df.copy()
for metric in metrics:
    max_val = radar_df[metric].max()
    radar_normalized[metric] = radar_df[metric] / max_val

labels = metrics
num_vars = len(labels)

# Criar o gráfico radial
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for i in range(len(radar_normalized)):
    values = radar_normalized.loc[i].drop('player').tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=radar_normalized.loc[i, 'player'])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title("Radar - Comparação Normalizada de Players", size=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.tight_layout()
plt.savefig("radar_comparacao_players.png")
plt.show()


